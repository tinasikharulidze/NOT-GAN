"""
Training-diagnostic and evaluation metrics.
"""

import torch
import numpy as np
import pandas as pd


def grad_norm(model):
    """Mean absolute gradient value across all trainable parameters.
    Smooth proxy for training stability - this is what the "Critic/Generator
    Gradient Norm" plots in the thesis actually show."""
    norms = [p.grad.abs().mean() for p in model.parameters() if p.grad is not None]
    return torch.tensor(0.0) if not norms else torch.stack(norms).mean()


def grad_sq_norm(model):
    """||grad L||^2 summed across all parameters of `model`.

    `grad_norm()` above is smoother for plotting; this is the quantity the
    theory actually makes claims about.
    """
    sq = [p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]
    return torch.tensor(0.0) if not sq else torch.stack(sq).sum()


def critic_output_grad_norm(D, y_samples):
    """Empirical Lipschitz proxy: mean ||grad_y D(y)||_2 across y_samples.

    The OT formulation (unlike WGAN) does not enforce 1-Lipschitz on D.
    This metric tracks what value the output-gradient norm actually settles
    at - directly answering "without enforcement, how Lipschitz is D?"
    """
    y = y_samples.detach().clone().requires_grad_(True)
    out = D(y).view(-1).sum()
    grads = torch.autograd.grad(out, y, create_graph=False)[0]
    return grads.norm(dim=1).mean()


def cluster_coverage(fake, centers):
    """Fraction of `fake` points whose nearest target center is each cluster.

    For a K-cluster target, returns a (K,) tensor: the histogram of "which
    cluster did each generated point fall closest to?". A healthy generator
    matches the target mixture weights; mode collapse shows up as one
    component near 1.0 and the others near 0.

    """
    d2 = (fake[:, None, :] - centers[None, :, :]).pow(2).sum(-1)  # (N, K)
    assignments = d2.argmin(-1)  # (N,)
    counts = torch.bincount(assignments, minlength=centers.shape[0])
    return counts.float() / fake.shape[0]


def grad_norm_l2(model):
    """Global L2 norm of the gradient across all trainable parameters:
    sqrt(sum_p ||grad_p||_2^2).

    NOTE: this is a *different quantity* from `grad_norm()` above (which is
    a mean-absolute-value proxy).
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def grad_norm_l2_tensor(model):
    """Same quantity as `grad_norm_l2` (global L2 norm of the gradient), but
    accumulated as a GPU tensor with no `.item()` calls -- avoids a
    host-device sync on every parameter. Used inside the CartoonSet
    training loop's hot path, where per-step syncs are expensive enough to
    matter; `grad_norm_l2` is fine everywhere else.
    """
    sq = None
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().pow(2).sum()
            sq = g if sq is None else sq + g
    if sq is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return sq ** 0.5


def compute_w2_squared(fake, real):
    """Exact squared Wasserstein-2 distance between two image batches.

    Solves the optimal-transport LP with squared-Euclidean cost (the same
    cost the OT-GAN transport term uses), with uniform weights (1/B).
    Computed on CPU via POT; well under a second per call at 256+256.

    fake, real: [B, C, H, W] tensors (any device).
    Returns: float -- W_2^2(G#p_eval, q_eval) in pixel space.

    NOTE: for CartoonSet (64x64x3 = 12,288-dim images), the exact LP this
    solves becomes too slow to run every epoch -- see
    `compute_w2_squared_sliced` below for the approximation actually used
    there. Both notebooks call their version "compute_w2_squared" in the
    originals; kept as separate functions here since they compute genuinely
    different things (exact vs. sliced/projected W2).
    """
    import ot

    A = fake.detach().flatten(1).cpu().numpy().astype(np.float64)
    B = real.flatten(1).cpu().numpy().astype(np.float64)
    a = np.full(len(A), 1.0 / len(A))
    b = np.full(len(B), 1.0 / len(B))
    M = ot.dist(A, B, metric='sqeuclidean')
    return float(ot.emd2(a, b, M))


def compute_w2_squared_sliced(fake, real, n_projections=200, seed=42):
    """Sliced-Wasserstein approximation of squared W2, used for CartoonSet
    (images are too high-dimensional at 64x64x3 for the exact LP in
    `compute_w2_squared` to be tractable every epoch). Averages the exact
    1D W2 distance over `n_projections` random directions -- an unbiased
    estimator of the true (non-squared) sliced-W2 distance, not identical
    to the exact W2 computed elsewhere in this codebase.
    """
    import ot

    A = fake.detach().flatten(1).cpu().float().numpy()
    B = real.flatten(1).cpu().float().numpy()
    return float(ot.sliced_wasserstein_distance(A, B, n_projections=n_projections, seed=seed))


def w2_to_target(fake, real, reg=0.5, max_n=500):
    """Sinkhorn-regularized squared W2 distance between two 2D point clouds.

    A pure evaluation metric - never used inside the training loss. Cost is
    squared L2. Uses log-domain Sinkhorn for numerical stability at small
    `reg`. Subsamples each side to `max_n` points to keep the n*n cost
    matrix cheap (~0.4s per call at n=500 in 2D, benchmarked on CPU).

    `reg` is the entropic regularization: smaller is closer to true W2 but
    stiffer numerically. The resulting bias is constant across runs at a
    fixed (reg, n), so this is fine for *ranking* experiments. For absolute
    W2 values, switch to `ot.emd2(a, b, M)` (exact, no bias, ~10x slower).
    """
    import ot

    n = min(max_n, fake.shape[0], real.shape[0])
    fi = torch.randperm(fake.shape[0], device=fake.device)[:n]
    ri = torch.randperm(real.shape[0], device=real.device)[:n]
    f_np = fake[fi].detach().cpu().numpy()
    r_np = real[ri].detach().cpu().numpy()
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = ot.dist(f_np, r_np, metric='sqeuclidean')
    return float(ot.sinkhorn2(a, b, M, reg, method='sinkhorn_log'))


# ============================================================================
# CartoonSet experiment -- CLIP embedding quality checks
# ============================================================================

def precompute_embeddings(df, model, preprocess, save_path='cartoon_clip_embeddings.npy',
                           batch_size=256, device='cuda'):
    """Runs the fine-tuned image encoder over every image once and saves the
    result to disk -- so GAN training never needs a CLIP forward pass, just
    an array lookup by row position.

    Returns: np.ndarray, shape [N, 512], L2-normalized.
    """
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from tqdm import tqdm

    model.eval()

    class _ImgDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.transform(Image.open(self.paths[i]).convert('RGB'))

    loader = DataLoader(_ImgDataset(df['img_path'].tolist(), preprocess),
                         batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    all_embs = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Pre-computing embeddings"):
            emb = model.encode_image(imgs.to(device))
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    np.save(save_path, embeddings)
    print(f"Saved {embeddings.shape}  dtype={embeddings.dtype}  -> {save_path}")
    return embeddings


def check_gap(df, model, preprocess, device='cuda', n=50):
    """Blonde-vs-dark-brown embedding similarity gap: a quick, fixed
    sanity check to run before/after CLIP fine-tuning to confirm the
    fine-tune actually improved attribute separability."""
    import torch.nn.functional as F
    from PIL import Image

    blonde = df[df['hair_color'] == 0].head(n)   # 0 = blonde
    dark = df[df['hair_color'] == 6].head(n)     # 6 = dark brown
    sample = pd.concat([blonde, dark]).reset_index(drop=True)
    labels = torch.tensor([0] * n + [1] * n)

    imgs = torch.stack([
        preprocess(Image.open(p).convert('RGB')) for p in sample['img_path']
    ]).to(device)

    model.eval()
    with torch.no_grad():
        embs = F.normalize(model.encode_image(imgs), dim=-1).cpu()

    sim = embs @ embs.T
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(len(labels)).bool()
    diff = (labels.unsqueeze(0) != labels.unsqueeze(1))

    print(f"Same-class similarity:  {sim[same].mean():.4f}")
    print(f"Cross-class similarity: {sim[diff].mean():.4f}")
    print(f"Gap:                    {sim[same].mean() - sim[diff].mean():.4f}")


def audit_all_attribute_gaps(df, model, preprocess, attributes, device='cuda', samples_per_class=15):
    """Runs `check_gap`-style same-vs-cross-class similarity for every
    attribute in `attributes`, and returns a DataFrame sorted by gap score
    (Table `clip_gap` in the thesis). Skips any attribute with fewer than 2
    classes or with no loadable images."""
    import torch.nn.functional as F
    from PIL import Image

    print("Initializing complete CLIP latent space audit...\n")
    model.eval()
    results = []

    for attr in attributes:
        if attr not in df.columns:
            print(f"Skipping '{attr}': column not found.")
            continue

        unique_classes = sorted(df[attr].dropna().unique())
        if len(unique_classes) < 2:
            print(f"Skipping '{attr}': needs >= 2 unique classes, found {len(unique_classes)}.")
            continue

        sampled_df_list = [df[df[attr] == c].head(samples_per_class) for c in unique_classes]
        audit_sample = pd.concat(sampled_df_list).reset_index(drop=True)
        labels = torch.tensor(audit_sample[attr].values, dtype=torch.long)

        img_tensors = []
        for p in audit_sample['img_path']:
            try:
                img_tensors.append(preprocess(Image.open(p).convert('RGB')))
            except Exception:
                continue  # skip corrupt files

        if len(img_tensors) < 2:
            print(f"Skipping '{attr}': no valid images could be processed.")
            continue

        imgs = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            embs = F.normalize(model.encode_image(imgs), dim=-1).cpu()

        sim_matrix = embs @ embs.T
        same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(len(labels)).bool()
        diff_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))

        same_sim = sim_matrix[same_mask].mean().item() if same_mask.any() else 0.0
        cross_sim = sim_matrix[diff_mask].mean().item() if diff_mask.any() else 0.0

        results.append({
            "Attribute": attr,
            "Unique Classes": len(unique_classes),
            "Same-Class Sim": same_sim,
            "Cross-Class Sim": cross_sim,
            "Gap Score": same_sim - cross_sim,
        })
        print(f"Audited '{attr}' successfully.")

    if not results:
        print("\nNo attributes were successfully audited. Check image paths / DataFrame content.")
        return pd.DataFrame()

    audit_df = pd.DataFrame(results).sort_values(by="Gap Score", ascending=False)

    display_df = audit_df.copy()
    for col in ["Same-Class Sim", "Cross-Class Sim", "Gap Score"]:
        display_df[col] = display_df[col].map('{:.4f}'.format)
    print("\n======================= METRIC GAP AUDIT SUMMARY =======================")
    print(display_df.to_string(index=False))
    print("========================================================================\n")

    return audit_df
