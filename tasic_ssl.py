import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
)
import scipy
import matplotlib.pyplot as plt


CONFIG = {
    "n_top_genes":   2000,
    "batch_size":    256,
    "embedding_dim": 128,   
    "projection_dim": 64,   
    "hidden_dim":    512,
    "temperature":   0.5,
    "epochs":        100,
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "mask_prob":     0.3,   
    "noise_std":     0.1,   
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "seed":          42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def load_tasic_data(path):
    data = pickle.load(open(path, "rb"))
    return data


def lognormalize_counts(tasic_dict):
    
    counts = tasic_dict['counts']
    
    # trying to catch all formats in which the counts might be loaded
    if scipy.sparse.issparse(counts):
        counts = counts.toarray()
    elif isinstance(counts, np.matrix):
        counts = np.squeeze(np.asarray(counts))
    else:
        raise TypeError(f"Data format is {type(counts)} but should be np.martix or a sparse matrix.")
    
    #normalize and logtransform counts
    libsizes = counts.sum(axis=1)
    CPM = counts / libsizes[:, None] * 1e+6
    
    logCPM = np.log2(CPM + 1) 
    tasic_dict['logCPM'] = logCPM  
    
    return tasic_dict


def preprocess(data):
    data = lognormalize_counts(data)
    return data


class SCAugmentations:
    """
    Two cheap augmentations that work well on scRNA-seq:
        - Random gene masking (mimics technical dropout)
        - Gaussian noise (mimics measurement noise)
    """
    def __init__(self, mask_prob=0.3, noise_std=0.1):
        self.mask_prob = mask_prob
        self.noise_std = noise_std

    def __call__(self, x):
        mask = (torch.rand_like(x) > self.mask_prob).float()
        x_aug = x * mask
        x_aug = x_aug + torch.randn_like(x_aug) * self.noise_std
        return x_aug


class ContrastiveSCDataset(Dataset):
    """Returns two augmented views of the same cell."""
    def __init__(self, X, augment):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return self.augment(x), self.augment(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SimCLRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, projection_dim):
        super().__init__()
        self.encoder   = Encoder(input_dim, hidden_dim, embedding_dim)
        self.projector = ProjectionHead(embedding_dim, projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (B, D) L2-normalized projections of two views.
    Positives: (i, i) pairs across z1/z2. Negatives: all other cells.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                       # (2B, D)
    sim = torch.mm(z, z.t()) / temperature               # (2B, 2B)

    # Remove self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive index for sample i is i+B (and vice versa)
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def train_ssl(model, loader, cfg):
    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    model.train()
    history = []
    for epoch in range(cfg["epochs"]):
        running = 0.0
        for x1, x2 in loader:
            x1 = x1.to(cfg["device"], non_blocking=True)
            x2 = x2.to(cfg["device"], non_blocking=True)

            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, cfg["temperature"])

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)

        sched.step()
        avg = running / len(loader.dataset)
        history.append(avg)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | loss={avg:.4f}")
    return history


@torch.no_grad()
def extract_embeddings(model, X, device, batch_size=512):
    model.eval()
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    out = []
    for i in range(0, X_t.size(0), batch_size):
        batch = X_t[i:i + batch_size].to(device)
        h, _ = model(batch)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_clustering(embeddings, labels, n_clusters=None):
    """Unsupervised: fit K-Means on embeddings, score against true labels."""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embeddings)
    ari = adjusted_rand_score(labels, km.labels_)
    nmi = normalized_mutual_info_score(labels, km.labels_)
    print(f"[Clustering]  K-Means    ARI={ari:.4f}  NMI={nmi:.4f}")
    return {"ari": ari, "nmi": nmi}


def evaluate_knn(embeddings, labels, k_values=(1, 5, 15), test_size=0.2, seed=0):
    """
    Standard SSL linear/kNN probe:
        Split embeddings into train/test, fit kNN on train, predict on test.
    Reports accuracy + macro-F1 (macro-F1 matters because TASIC cell types are
    very imbalanced — rare types would otherwise be hidden by accuracy).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels,
        test_size=test_size, 
        # stratify=labels, 
        random_state=seed,
    )
    results = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        results[k] = {"acc": acc, "macro_f1": f1m}
        print(f"[kNN probe]   k={k:<3d}    acc={acc:.4f}  macro-F1={f1m:.4f}")
    return results


def visualize(embeddings, labels, prefix="tasic_ssl"):
    """Project embeddings to 2-D with both UMAP and t-SNE, save side-by-side."""
    # t-SNE
    print("Running t-SNE…")
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=0)
    emb_tsne = tsne.fit_transform(embeddings)

    emb_umap = None
    try:
        import umap
        print("Running UMAP…")
        emb_umap = umap.UMAP(random_state=0).fit_transform(embeddings)
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot.")

    n_panels = 2 if emb_umap is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    axes[0].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                    c=labels, s=3, cmap="tab20")
    axes[0].set_title("t-SNE of SSL embeddings")
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    if emb_umap is not None:
        axes[1].scatter(emb_umap[:, 0], emb_umap[:, 1],
                        c=labels, s=3, cmap="tab20")
        axes[1].set_title("UMAP of SSL embeddings")
        axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

    fig.suptitle("TASIC — SimCLR self-supervised embeddings")
    plt.tight_layout()
    out_path = f"{prefix}_2d.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved 2-D projections to {out_path}")


def main(data_path, label_key="cell_type"):
    cfg = CONFIG

    print("Loading TASIC…")
    data = load_tasic_data(data_path)

    data = preprocess(data)

    X = data['logCPM'].toarray() if hasattr(data['logCPM'], "toarray") else data['logCPM']

    loader = DataLoader(
        ContrastiveSCDataset(X, SCAugmentations(cfg["mask_prob"], cfg["noise_std"])),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )

    model = SimCLRModel(
        input_dim=X.shape[1],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        projection_dim=cfg["projection_dim"],
    ).to(cfg["device"])

    print("Pre-training with contrastive SSL…")
    train_ssl(model, loader, cfg)

    print("Extracting embeddings…")
    embeddings = extract_embeddings(model, X, cfg["device"])
    np.save("tasic_ssl_embeddings.npy", embeddings)

    if label_key in data:
        labels = data[label_key]
        evaluate_clustering(embeddings, labels)
        evaluate_knn(embeddings, labels)
        visualize(embeddings, labels)
    else:
        print(f"No '{label_key}' column in data; skipping evaluation.")

    torch.save(model.state_dict(), "tasic_ssl_model.pt")
    print("Saved model weights to tasic_ssl_model.pt")
    return model, embeddings


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tasic.h5ad"
    key  = sys.argv[2] if len(sys.argv) > 2 else "clusters"
    main(path, key)