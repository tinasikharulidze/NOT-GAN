"""
Matplotlib figure builders for the 2D experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================================
# 2D toy experiment
# ============================================================================

def plot_source_target(source_points, target_points, title):
    """Scatter plot of the raw source and target point clouds."""
    plt.figure(figsize=(8, 5))
    plt.scatter(source_points[:, 0], source_points[:, 1],
                color='dodgerblue', alpha=0.1, s=10, label='Source Distribution $p \\in \\Delta_X$',
                edgecolors='navy', linewidths=1)
    plt.scatter(target_points[:, 0], target_points[:, 1],
                color='crimson', alpha=0.1, s=10, label='Target Distribution $q \\in \\Delta_Y$',
                edgecolors='darkred', linewidths=1)
    plt.axhline(0, color='black', lw=1, alpha=0.3)
    plt.axvline(0, color='black', lw=1, alpha=0.3)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_transported_state(real, fake, source, epoch):
    """Source / real-target / generated points overlaid at a given epoch."""
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(source[:, 0], source[:, 1], c='dodgerblue', alpha=0.2,
                label='Source (p)', s=10, edgecolors='navy', linewidths=1)
    plt.scatter(real[:, 0], real[:, 1], c='crimson', alpha=0.2,
                label='Target (q)', s=10, edgecolors='darkred', linewidths=1)
    plt.scatter(fake[:, 0], fake[:, 1], c='orange', alpha=0.2,
                label='Generated (pi)', s=10, edgecolors='darkorange', linewidths=1)
    plt.title(f"Transported Points at Epoch {epoch}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_transport_arrows(source, fake, epoch, n=200, seed=0):
    """Arrows from a random subset of source points to where the generator
    sent them - a direct visualization of the learned transport map."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(source), generator=g)[:n]
    s, f = source[idx].cpu(), fake[idx].cpu()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.quiver(s[:, 0], s[:, 1], f[:, 0] - s[:, 0], f[:, 1] - s[:, 1],
              angles='xy', scale_units='xy', scale=1,
              alpha=0.4, width=0.003, color='gray')
    ax.scatter(s[:, 0], s[:, 1], s=15, c='dodgerblue',
               label='source', edgecolors='navy', linewidths=0.5)
    ax.scatter(f[:, 0], f[:, 1], s=15, c='orange',
               label='fake', edgecolors='darkorange', linewidths=0.5)
    ax.set_title(f'Transport map - epoch {epoch}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')
    return fig


# ============================================================================
# MNIST experiment
# ============================================================================

def plot_generated_images(real, fake, epoch, seed, n=4):
    """Top row: n real/source images. Bottom row: the matching generated
    images. `seed` controls which real images are sampled (kept fixed
    across an epoch so successive figures are visually comparable)."""
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(real.size(0), generator=rng)[:n]
    real = real[idx]
    fake = fake[:n]

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(real[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(fake[i, 0].cpu().clamp(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    axes[0, 0].set_title('Input', fontsize=8)
    axes[1, 0].set_title(f'Output (Epoch: {epoch})', fontsize=8)
    fig.tight_layout()
    return fig


def plot_conditional_grid(G, sample_noise, ref_real, epoch, device,
                           num_classes=10, samples_per_class=8):
    """Grid of generated samples: rows = digit class, cols = samples per class."""
    G.eval()
    with torch.no_grad():
        y = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
        ref = ref_real[:1].repeat(len(y), 1, 1, 1)
        x = sample_noise(ref)
        fake = G(x, y).cpu()

    fig, axes = plt.subplots(num_classes, samples_per_class,
                              figsize=(samples_per_class * 1.0, num_classes * 1.0))
    for r in range(num_classes):
        for c in range(samples_per_class):
            ax = axes[r, c]
            ax.imshow(fake[r * samples_per_class + c, 0].clamp(0, 1),
                       cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(str(r), rotation=0, fontsize=10, labelpad=10, va='center')
    fig.suptitle(f'Conditional samples — Epoch {epoch}', fontsize=10)
    fig.tight_layout()
    G.train()
    return fig


# ============================================================================
# CartoonSet experiment
# ============================================================================

def plot_grid_per_class(df, feature_name, constraint_mask, num_samples_per_class=3, title_prefix=""):
    """Dataset-exploration grid: filters `df` by `constraint_mask`, groups by
    `feature_name`, and plots one column per distinct class with
    `num_samples_per_class` random example rows. `constraint_mask` narrows
    the sample to rows where every *other* varying attribute is held fixed,
    so the grid isolates the effect of `feature_name` alone. Assumes `df`
    has a working `img_path` column (as built by the dataset-loading cell)."""
    from PIL import Image

    filtered_df = df[constraint_mask]
    distinct_classes = sorted(filtered_df[feature_name].unique())
    num_classes = len(distinct_classes)

    if num_classes == 0:
        print(f"No samples found for feature '{feature_name}' with the given filters.")
        return

    fig, axes = plt.subplots(num_samples_per_class, num_classes,
                              figsize=(num_classes * 2.5, num_samples_per_class * 2.5))

    if num_classes == 1 and num_samples_per_class == 1:
        axes = np.array([[axes]])
    elif num_classes == 1:
        axes = np.expand_dims(axes, axis=1)
    elif num_samples_per_class == 1:
        axes = np.expand_dims(axes, axis=0)

    for col_idx, class_val in enumerate(distinct_classes):
        class_rows = filtered_df[filtered_df[feature_name] == class_val]
        n_to_sample = min(num_samples_per_class, len(class_rows))
        sampled_rows = class_rows.sample(n=n_to_sample)

        for row_idx in range(num_samples_per_class):
            ax = axes[row_idx, col_idx]
            if row_idx < len(sampled_rows):
                row_data = sampled_rows.iloc[row_idx]
                try:
                    ax.imshow(Image.open(row_data['img_path']))
                except Exception:
                    ax.text(0.5, 0.5, "Missing\nImg", ha='center', va='center', color='red', fontsize=8)
            else:
                ax.text(0.5, 0.5, "Empty\nSlot", ha='center', va='center', color='gray', fontsize=8)

            if row_idx == 0:
                ax.set_title(f"Class: {class_val}", fontsize=10, fontweight='bold')
            ax.axis('off')

    fig.suptitle(f"{title_prefix} | Feature: [{feature_name}]", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.show()
    return fig


def plot_clip_clusters(attribute_name, target_ids, display_labels, hex_colors, coordinates):
    """2D UMAP scatter of CLIP embeddings, colored by a categorical
    attribute (Appendix C of the thesis)."""
    fig = plt.figure(figsize=(11, 8))

    for idx in sorted(display_labels.keys()):
        mask = (target_ids == idx)
        if np.any(mask):
            plt.scatter(coordinates[mask, 0], coordinates[mask, 1],
                        c=hex_colors[idx], label=display_labels[idx],
                        s=4, alpha=0.65, edgecolors='none')

    plt.title(f'CLIP Latent Space Cluster Analysis — {attribute_name}', fontsize=14, pad=15)
    plt.xlabel('UMAP Dimension 1', fontsize=10)
    plt.ylabel('UMAP Dimension 2', fontsize=10)
    plt.legend(title=f"{attribute_name} Categories", markerscale=4, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()
    return fig
