"""
Sampling and subsampling helpers for the source/target distributions.
"""

import torch
import numpy as np
import pandas as pd


# ============================================================================
# 2D toy experiment
# ============================================================================

def sample_gaussian(n, mu, cov_matrix, seed=None):
    """Sample n points from a 2D Multivariate Normal."""
    if seed is not None:
        torch.manual_seed(seed)
    mu = torch.as_tensor(mu, dtype=torch.float32)
    cov = torch.as_tensor(cov_matrix, dtype=torch.float32)
    return torch.distributions.MultivariateNormal(mu, cov).sample((n,))


def sample_mog(n_total, means, covs, weights, seed=None):
    """Sample from a Mixture of Gaussians, shuffled at the end.

    Cluster sample counts are allocated by largest-remainder rounding, so
    they sum to exactly `n_total` even when `weights * n_total` isn't
    integer.
    """
    if seed is not None:
        torch.manual_seed(seed)
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()
    n_clusters = len(means)

    samples_per_cluster = torch.floor(weights * n_total).int()
    diff = n_total - samples_per_cluster.sum()
    if diff > 0:
        fractional = (weights * n_total) - torch.floor(weights * n_total)
        _, idx = torch.topk(fractional, diff)
        samples_per_cluster[idx] += 1

    all_samples = []
    for i in range(n_clusters):
        n_i = samples_per_cluster[i].item()
        if n_i > 0:
            all_samples.append(sample_gaussian(n_i, means[i], covs[i]))
    samples = torch.cat(all_samples, dim=0)

    perm = torch.randperm(samples.shape[0])
    return samples[perm]


def random_subset(points, n=10_000, seed=None):
    """Subsample n points. Uses a local generator so the global RNG is
    never mutated by a plotting/logging call."""
    total = points.shape[0]
    n = min(n, total)
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(total, generator=g)
    else:
        idx = torch.randperm(total)
    return points[idx[:n]]


# ============================================================================
# MNIST experiment
# ============================================================================

def get_mnist_loaders(batch_size=128, subset_size=2000, seed=42, img_size=32,
                       label=None, device='cpu'):
    """Load MNIST, optionally filtered to a single digit class and/or
    subsampled to `subset_size` training images. Returns (train_loader,
    test_loader); the test set is always the full (unfiltered-by-subset)
    MNIST test split, optionally filtered to `label`.
    """
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if label is not None:
        train_mask = (train_set.targets == label)
        test_mask = (test_set.targets == label)
        train_set = Subset(train_set, torch.where(train_mask)[0])
        test_set = Subset(test_set, torch.where(test_mask)[0])

    full_set = train_set
    if subset_size is not None:
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(full_set), subset_size, replace=False)
        full_set = Subset(full_set, indices)

    kwargs = dict(batch_size=batch_size, pin_memory=(device == 'cuda'),
                  num_workers=4, persistent_workers=True)
    train_loader = DataLoader(full_set, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, test_loader


# ============================================================================
# CartoonSet experiment -- CLIP fine-tuning dataset
# ============================================================================

class CartoonCLIPDataset(torch.utils.data.Dataset):
    """Returns (image_tensor, token_ids) pairs for CLIP contrastive
    fine-tuning. `df` needs an `img_path` column and a `clip_text_description`
    column (see `captions.generate_clip_caption`)."""

    def __init__(self, df, tokenizer, transform):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        img = Image.open(row['img_path']).convert('RGB')
        img = self.transform(img)
        text = self.tokenizer([row['clip_text_description']])[0]  # [77]
        return img, text


# ============================================================================
# CartoonSet experiment -- GAN training dataset (distinct from
# `CartoonCLIPDataset` above: this one pairs each image with its
# *precomputed* CLIP embedding rather than a tokenized caption, since GAN
# training conditions on the embedding directly and never runs CLIP itself).
# ============================================================================

class CartoonGANDataset(torch.utils.data.Dataset):
    """Returns (image_tensor, clip_embedding) pairs. Preloads and resizes
    every image into RAM up front (`preload=True`) -- CartoonSet is small
    enough at 64x64 that this is faster than re-decoding from disk every
    epoch."""

    def __init__(self, df, embeddings, transform, img_size=64, preload=True):
        from torchvision import transforms as T

        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.transform = transform
        self.images = None

        if preload:
            from PIL import Image
            print("Pre-loading resized images into RAM...")
            self.images = []
            resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)
            for _, row in self.df.iterrows():
                img = Image.open(row['img_path']).convert('RGB')
                self.images.append(resize(img).copy())
            print(f"Done. ~{len(self.images) * img_size * img_size * 3 / 1e6:.0f} MB in RAM")

    def __getitem__(self, idx):
        from PIL import Image
        img = (self.images[idx] if self.images is not None
               else Image.open(self.df.iloc[idx]['img_path']).convert('RGB'))
        return self.transform(img), torch.tensor(
            self.embeddings[int(self.df.iloc[idx]['emb_idx'])], dtype=torch.float32)

    def __len__(self):
        return len(self.df)


def get_clip_loaders(metadata_csv, embeddings_path, images_dir,
                      train_size=50_000, test_size=1_000, img_size=64,
                      batch_size=128, num_workers=2, prefetch_factor=2):
    """Loads the preprocessed CartoonSet CSV (see
    `CartoonSet_Preprocessing.ipynb`) and precomputed CLIP embeddings, and
    builds train/test `CartoonGANDataset` loaders.

    `images_dir` replaces whatever path prefix the CSV's `img_path` column
    was originally saved with -- i.e. point it at wherever you actually have
    the cartoonset100k images extracted on *this* machine.
    """
    from torchvision import transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.ToTensor()])  # resize already done in preload

    df = pd.read_csv(metadata_csv)
    df['img_path'] = df['img_path'].str.replace(
        '/kaggle/working/cartoonset100k', images_dir, regex=False)
    embeddings = np.load(embeddings_path)  # [N, 512]

    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:(train_size + test_size)].reset_index(drop=True)

    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        CartoonGANDataset(train_df, embeddings, transform, img_size=img_size),
        batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    test_loader = DataLoader(
        CartoonGANDataset(test_df, embeddings, transform, img_size=img_size),
        batch_size=batch_size, shuffle=False, **kw)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_loader, test_loader, train_df, test_df, embeddings
