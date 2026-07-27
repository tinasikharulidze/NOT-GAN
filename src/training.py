"""
Training loops. One function per experiment family, since the loops differ
structurally (conditioning, CLIP embeddings, gradient penalties, ...) even
though they share the same primal-dual update rules from `optimizers.py`.
"""

import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import (
    grad_norm, grad_norm_l2, grad_norm_l2_tensor, grad_sq_norm,
    cluster_coverage, compute_w2_squared, compute_w2_squared_sliced,
)
from optimizers import adam_update, optimistic_adam_update, anchored_adam_update, anchored_optimistic_adam_update
from plotting import (
    plot_transported_state, plot_transport_arrows,
    plot_generated_images, plot_conditional_grid,
)
from data import random_subset, CartoonCLIPDataset, CartoonGANDataset, get_clip_loaders


# ============================================================================
# 2D toy experiment
# ============================================================================

def train_2d(
    G, D,
    source_points, target_points,
    optimizer_fn,
    gamma, eta, sigma,
    num_epochs=200, batch_size=32,
    subset_size=5000, seed=42,
    run_name='run',
    target_centers=None,
    viz_every=20,
    device='cpu',
    log_dir='runs/',
    model_dir='models/',
):
    """Primal-dual OT training loop for the 2D toy experiment.

    `gamma`/`eta` (critic/generator step sizes) accept either a plain float
    or a schedule function `lr_fn(step)` from `optimizers.py`.
    """
    torch.manual_seed(seed)

    G = G.to(device)
    D = D.to(device)
    source_points = source_points.to(device)
    target_points = target_points.to(device)
    if target_centers is not None:
        target_centers = target_centers.to(device)

    writer = SummaryWriter(f'{log_dir}/{run_name}')

    num_batches = len(source_points) // batch_size
    print(f"Number of steps per epoch: {num_batches}")
    print(f"Number of steps in total: {num_batches * num_epochs}")
    print(f"device={device}, seed={seed}")

    d_moments = {}
    g_moments = {}
    update_step = 1

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        src_perm = torch.randperm(len(source_points), device=device)
        tgt_perm = torch.randperm(len(target_points), device=device)

        # Accumulate as GPU tensors so there's no .item() sync inside the batch loop.
        acc = {k: torch.zeros(1, device=device) for k in [
            'loss_D', 'loss_G', 'real_score', 'fake_score',
            'D_grad', 'G_grad', 'D_grad_sq', 'G_grad_sq',
            'D_real_var', 'D_fake_var', 'G_out_var', 'd_fake_logged', 'c',
        ]}
        acc_lr_D = 0.0
        acc_lr_G = 0.0

        for t in range(num_batches):
            idx_s = src_perm[t * batch_size:(t + 1) * batch_size]
            idx_t = tgt_perm[t * batch_size:(t + 1) * batch_size]

            X_t = source_points[idx_s]
            Y_t = target_points[idx_t]

            # Reparametrization trick: Yhat = G(X) + sigma * Z, Z ~ N(0, I).
            tmp = G(X_t)
            z = torch.randn_like(tmp)
            Yhat_t = tmp + sigma * z

            # --- Dual (critic) update: maximize E[f(Y)] - E[f(Yhat)] ---
            Yhat_detached = Yhat_t.detach()
            real_out = D(Y_t)
            fake_out = D(Yhat_detached)
            real_score = real_out.view(-1).mean()
            fake_score = fake_out.view(-1).mean()
            loss_D = real_score - fake_score

            D.zero_grad()
            loss_D.backward()
            D_grad = grad_norm(D)
            D_grad_sq = grad_sq_norm(D)
            gamma_t = gamma(update_step) if callable(gamma) else gamma
            optimizer_fn(D.parameters(), d_moments, update_step, lr=gamma_t, is_discriminator=True)
            D.zero_grad()

            # --- Primal (generator) update: minimize transport cost - critic score ---
            c = ((Yhat_t - X_t) ** 2).sum(dim=1)
            loss_G = (c - D(Yhat_t).view(-1)).mean()

            G.zero_grad()
            loss_G.backward()
            G_grad = grad_norm(G)
            G_grad_sq = grad_sq_norm(G)
            eta_t = eta(update_step) if callable(eta) else eta
            optimizer_fn(G.parameters(), g_moments, update_step, lr=eta_t, is_discriminator=False)
            G.zero_grad()

            update_step += 1

            with torch.no_grad():
                d_fake_logged = D(Yhat_t).view(-1).mean()

            acc['loss_D'] += loss_D.detach()
            acc['loss_G'] += loss_G.detach()
            acc['real_score'] += real_score.detach()
            acc['fake_score'] += fake_score.detach()
            acc['D_grad'] += D_grad.detach()
            acc['D_grad_sq'] += D_grad_sq.detach()
            acc['G_grad'] += G_grad.detach()
            acc['G_grad_sq'] += G_grad_sq.detach()
            acc['D_real_var'] += real_out.var().detach()
            acc['D_fake_var'] += fake_out.var().detach()
            acc['G_out_var'] += Yhat_t.var().detach()
            acc['d_fake_logged'] += d_fake_logged.detach()
            acc['c'] += c.mean().detach()
            acc_lr_D += gamma_t
            acc_lr_G += eta_t

        # One .item() call per stat, once per epoch.
        avg = {k: (v / num_batches).item() for k, v in acc.items()}
        avg['lr_D'] = acc_lr_D / num_batches
        avg['lr_G'] = acc_lr_G / num_batches

        writer.add_scalar("Critic Loss", avg['loss_D'], epoch)
        writer.add_scalar("Generator Loss", avg['loss_G'], epoch)
        writer.add_scalar("Critic Score/Real Samples", avg['real_score'], epoch)
        writer.add_scalar("Critic Score/Generated Samples", avg['fake_score'], epoch)
        writer.add_scalar("Critic Score/Gap", avg['real_score'] - avg['fake_score'], epoch)
        writer.add_scalar("Critic Gradient Norm", avg['D_grad'], epoch)
        writer.add_scalar("Generator Gradient Norm", avg['G_grad'], epoch)
        writer.add_scalar("Var/D_real", avg['D_real_var'], epoch)
        writer.add_scalar("Var/D_fake", avg['D_fake_var'], epoch)
        writer.add_scalar("Var/G_output", avg['G_out_var'], epoch)
        writer.add_scalar("Transport Cost", avg['c'], epoch)
        writer.add_scalar("Ratio/C_over_D", avg['c'] / (abs(avg['d_fake_logged']) + 1e-8), epoch)
        writer.add_scalar("Learning Rate/gamma", avg['lr_D'], epoch)
        writer.add_scalar("Learning Rate/eta", avg['lr_G'], epoch)

        if target_centers is not None:
            with torch.no_grad():
                cov_idx = torch.randperm(len(source_points), device=device)[:2_000]
                fake_sample_full = G(source_points[cov_idx])
            coverage = cluster_coverage(fake_sample_full, target_centers)
            for k, frac in enumerate(coverage):
                writer.add_scalar(f"Coverage/cluster_{k}", frac, epoch)

        if epoch % viz_every == 0:
            with torch.no_grad():
                current_fake = G(source_points)

            real_sample = random_subset(target_points, subset_size, seed=seed).cpu()
            fake_sample = random_subset(current_fake, subset_size, seed=seed).cpu()
            src_sample = random_subset(source_points, subset_size, seed=seed).cpu()

            fig = plot_transported_state(real=real_sample, fake=fake_sample,
                                          source=src_sample, epoch=epoch)
            writer.add_figure("Visuals/Transported_State", fig, global_step=epoch)
            plt.close(fig)

            fig = plot_transport_arrows(source=src_sample, fake=fake_sample, epoch=epoch)
            writer.add_figure("Visuals/Transport_Arrows", fig, global_step=epoch)
            plt.close(fig)

    gamma_log = getattr(gamma, 'initial_lr', gamma) if callable(gamma) else gamma
    eta_log = getattr(eta, 'initial_lr', eta) if callable(eta) else eta
    hparams = {
        'lr_D (gamma)': gamma_log,
        'lr_G (eta)': eta_log,
        'lr_schedule_D': getattr(gamma, 'name', 'constant'),
        'lr_schedule_G': getattr(eta, 'name', 'constant'),
        'batch_size': batch_size,
        'sigma': sigma,
        'num_epochs': num_epochs,
        'optimizer': getattr(optimizer_fn, '__name__', str(optimizer_fn)),
        'seed': seed,
    }
    final_metrics = {
        'final/loss_G': avg['loss_G'],
        'final/loss_D': avg['loss_D'],
        'final/D_gap': avg['real_score'] - avg['fake_score'],
        'final/C': avg['c'],
    }
    if target_centers is not None:
        for k, frac in enumerate(coverage):
            final_metrics[f'final/coverage_{k}'] = frac
    writer.add_hparams(hparams, final_metrics)

    os.makedirs(f'{model_dir}/{run_name}', exist_ok=True)
    ckpt_meta = {'seed': seed, 'hparams': hparams, 'epoch': num_epochs, 'run_name': run_name}
    torch.save({'state_dict': G.state_dict(), **ckpt_meta}, f'{model_dir}/{run_name}/G.pt')
    torch.save({'state_dict': D.state_dict(), **ckpt_meta}, f'{model_dir}/{run_name}/D.pt')

    writer.close()
    return G, D


# ============================================================================
# MNIST experiment -- shared logging helpers
#
# Used by both `train_mnist` (unconditional) and `train_mnist_cond`
# (class-conditional) below.
# ============================================================================

def _resolve_gamma(gamma_spec, epoch):
    """Return the critic step size to use at this epoch.

    gamma_spec can be:
      - a scalar (constant gamma for the whole run), or
      - a 3-tuple (gamma_initial, gamma_after, drop_epoch): use gamma_initial
        for epoch <= drop_epoch, then gamma_after.
    """
    if isinstance(gamma_spec, (tuple, list)):
        gamma_initial, gamma_after, drop_epoch = gamma_spec
        return gamma_initial if epoch <= drop_epoch else gamma_after
    return gamma_spec


def _log_hparams(writer, G, D, gamma, eta, sigma, k_critic, k_generator,
                  warmup_steps, warmup_k_critic, beta1, optimizer_fn,
                  initial_noise_type, run_name):
    hparams = {
        "run_name": run_name,
        "run_id": run_name.split('/')[0],
        "Generator_name": G.__class__.__name__,
        "Discriminator_name": D.__class__.__name__,
        "gamma": gamma,
        "eta": eta,
        "sigma": sigma,
        "k_critic": k_critic,
        "k_generator": k_generator,
        "warmup_steps": warmup_steps,
        "warmup_k_critic": warmup_k_critic,
        "b1": beta1,
        "optimizer": optimizer_fn.__name__,
        "initial_noise": initial_noise_type,
    }
    writer.add_hparams(hparam_dict=hparams, metric_dict={"init_metric": 0.0})


def _log_epoch_scalars(writer, avg, g_drift, diversity, epoch):
    writer.add_scalar("Collapse/G_param_drift_from_init", g_drift, epoch)
    writer.add_scalar("Collapse/G_output_diversity", diversity, epoch)
    writer.add_scalar("Critic Loss", avg['loss_D'], epoch)
    writer.add_scalar("Generator Loss", avg['loss_G'], epoch)
    writer.add_scalar("Critic Score on Y", avg['real_score'], epoch)
    writer.add_scalar("Critic Score on Ŷ", avg['fake_score'], epoch)
    writer.add_scalar("Critic Gradient Norm", avg['D_grad'], epoch)
    writer.add_scalar("Generator Gradient Norm", avg['G_grad'], epoch)
    writer.add_scalar("Var/D_real", avg['D_real_var'], epoch)
    writer.add_scalar("Var/D_fake", avg['D_fake_var'], epoch)
    writer.add_scalar("Variance of Ŷ", avg['G_out_var'], epoch)
    writer.add_scalar("Transport Cost", avg['c'], epoch)
    writer.add_scalar("Cost/Critic_Score for Ŷ", avg['c'] / (abs(avg['d_fake_logged']) + 1e-8), epoch)
    writer.add_scalar("Identity Gap", avg['identity_gap'], epoch)


def _log_visuals(writer, G, sample_noise, initial_noise_type, Y_real, epoch, device, img_size=32):
    G.eval()
    with torch.no_grad():
        fixed_x = sample_noise(Y_real)
        fixed_fake_img = G(fixed_x).cpu()
        fig = plot_generated_images(fixed_x, fixed_fake_img, epoch, seed=epoch)
        writer.add_figure("Visuals/Generated", fig, global_step=epoch)
        plt.close(fig)

    with torch.no_grad():
        if initial_noise_type == "uniform":
            X_t = torch.rand(64, 1, img_size, img_size, device=device)
        elif initial_noise_type == "low_freq":
            X_t = F.interpolate(
                torch.rand(64, 1, 8, 8, device=device),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False,
            )
        else:
            raise ValueError("incorrect initial_noise_type")

        e1 = G.enc1(X_t)
        e2 = G.enc2(e1)
        e3 = G.enc3(e2)
        e4 = G.enc4(e3)
        e5 = G.enc5(e4)

        writer.add_scalar("Var of activations on e1", e1.var(dim=0).mean().item(), epoch)
        writer.add_scalar("Var of activations on e2", e2.var(dim=0).mean().item(), epoch)
        writer.add_scalar("Var of activations on e3", e3.var(dim=0).mean().item(), epoch)
        writer.add_scalar("Var of activations on e4", e4.var(dim=0).mean().item(), epoch)
        writer.add_scalar("Var of activations on e5", e5.var(dim=0).mean().item(), epoch)
    G.train()


def _log_umap(writer, G, test_loader, epoch, device):
    """UMAP projection of the generator's bottleneck features, colored by
    digit class. Requires the `umap-learn` package."""
    import umap

    G.eval()
    embeddings, labels_list = [], []
    # CondGenerator's enc1 expects [B, channels+1, H, W] because it concats a
    # class map onto the input. Detect that and build the class map here.
    is_cond = hasattr(G, 'class_emb')

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            if is_cond:
                lbls_d = lbls.to(device)
                cls_map = G.class_emb(lbls_d).view(lbls_d.size(0), 1, G.img_size, G.img_size)
                x_in = torch.cat([imgs, cls_map], dim=1)
            else:
                x_in = imgs
            e = G.enc5(G.enc4(G.enc3(G.enc2(G.enc1(x_in)))))  # [B, 512, 1, 1]
            embeddings.append(e.view(e.size(0), -1).cpu().numpy())
            labels_list.append(lbls.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels_list)

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=4, alpha=0.7)
    fig.colorbar(scatter, ax=ax, ticks=range(10))
    ax.set_title(f'Bottleneck UMAP — Epoch {epoch}')
    writer.add_figure("Visuals/Bottleneck_UMAP", fig, global_step=epoch)
    plt.close(fig)
    G.train()


# ============================================================================
# MNIST experiment -- unconditional (Section 5.2-5.3 of the thesis)
# ============================================================================

def train_mnist(G, D, optimizer_fn, train_loader,
                 gamma=0.0001, eta=0.0001, sigma=0.001, beta1=0.5, beta2=0.9,
                 num_epochs=200, anchor_power=1, seed=42, run_name='run', device='cpu',
                 k_critic=5, k_generator=1,
                 warmup_steps=80, warmup_k_critic=10,
                 log_every=50, initial_noise_type="uniform",
                 resume_G_path=None, resume_D_path=None,
                 w2_eval_size=256,
                 ema_decay=None,
                 reset_every=10,
                 img_size=32,
                 log_dir='MNIST_runs', model_dir='models'):
    """Primal-dual OT training loop for unconditional MNIST.

    `gamma` accepts either a constant float or a 3-tuple
    `(gamma_initial, gamma_after, drop_epoch)` for a one-time step-down
    schedule (see `_resolve_gamma`). Checkpoints are written every
    `log_every` epochs (default matches the original notebook's every-50)
    to `{model_dir}/{run_name}_G_ep{epoch}.pth`, plus a final checkpoint at
    the end of training.
    """
    writer = SummaryWriter(f'{log_dir}/{run_name}')
    num_batches = len(train_loader)
    print(f"Steps/epoch: {num_batches}  |  Total steps: {num_batches * num_epochs}")
    if isinstance(gamma, (tuple, list)):
        gi, ga, de = gamma
        print(f"Stepped gamma: {gi} for epochs 1..{de}, then {ga} for epochs {de + 1}..{num_epochs}")

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 1
    update_step = 1
    if resume_G_path is not None:
        G.load_state_dict(torch.load(resume_G_path, map_location=device))
        D.load_state_dict(torch.load(resume_D_path, map_location=device))
        # parse epoch number from filename e.g. "models/.../run_G_ep50.pth" -> 50
        start_epoch = int(re.search(r'_ep(\d+)\.pth', resume_G_path).group(1)) + 1
        update_step = (start_epoch - 1) * num_batches + 1
        print(f"Resumed from epoch {start_epoch - 1}  |  update_step set to {update_step}")

    # ── EMA generator (post-hoc smoother; doesn't touch the optimizer) ──
    if ema_decay is not None:
        G_ema = deepcopy(G)
        for p in G_ema.parameters():
            p.requires_grad_(False)
        G_ema.eval()
        print(f"EMA enabled: decay={ema_decay}  (effective window ≈ {int(1 / (1 - ema_decay))} G-steps)")
    else:
        G_ema = None

    critic_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                shuffle=True, num_workers=train_loader.num_workers)
    critic_iter = iter(critic_loader)

    def get_batch():
        nonlocal critic_iter
        try:
            return next(critic_iter)
        except StopIteration:
            critic_iter = iter(critic_loader)
            return next(critic_iter)

    def make_noise_fn(noise_type):
        if noise_type == "uniform":
            return lambda ref: torch.rand_like(ref)
        elif noise_type == "low_freq":
            return lambda ref: F.interpolate(
                torch.rand(ref.size(0), ref.size(1), 8, 8, device=device),
                size=(ref.size(2), ref.size(3)),
                mode='bilinear',
                align_corners=False,
            )
        else:
            raise ValueError("incorrect initial_noise_type")

    sample_noise = make_noise_fn(initial_noise_type)

    # Fixed noise batch -- same every epoch so diversity is comparable across epochs.
    torch.manual_seed(seed)
    fixed_noise = torch.rand(64, 1, img_size, img_size, device=device)

    # Fixed eval batches for W²(G#p, q) -- sampled once, reused every epoch.
    # Larger than fixed_noise (256 vs 64) so the LP estimate is stable. RNG
    # state is saved and restored so training stochasticity is unaffected.
    saved_rng = torch.get_rng_state()
    eval_loader = DataLoader(train_loader.dataset, batch_size=w2_eval_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 1))
    fixed_eval_real, _ = next(iter(eval_loader))
    fixed_eval_real = fixed_eval_real.to(device)
    torch.manual_seed(seed + 2)
    fixed_eval_noise = sample_noise(fixed_eval_real).detach()
    torch.set_rng_state(saved_rng)

    # Snapshot initial G parameters for drift tracking.
    init_params = {i: p.data.clone() for i, p in enumerate(G.parameters())}

    d_moments = {}
    g_moments = {}

    prev_gamma = None
    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs"):
        gamma_now = _resolve_gamma(gamma, epoch)
        if prev_gamma is not None and gamma_now != prev_gamma:
            print(f"[epoch {epoch}] gamma stepped: {prev_gamma} -> {gamma_now}")
        prev_gamma = gamma_now
        writer.add_scalar("Schedule/gamma", gamma_now, epoch)

        if epoch == start_epoch:
            _log_hparams(writer, G, D, gamma_now, eta, sigma, k_critic, k_generator,
                         warmup_steps, warmup_k_critic, beta1, optimizer_fn,
                         initial_noise_type, run_name)

        acc = {k: torch.zeros(1, device=device) for k in [
            'loss_D', 'loss_G', 'real_score', 'fake_score',
            'D_grad', 'G_grad', 'D_real_var', 'D_fake_var',
            'G_out_var', 'd_fake_logged', 'c', 'identity_gap',
        ]}

        for Y_t, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, total=num_batches):
            Y_t = Y_t.to(device)
            k_c = warmup_k_critic if update_step < warmup_steps else k_critic

            for _ in range(k_c):
                Y_real, _ = get_batch()
                Y_real = Y_real.to(device)
                X_t_c = sample_noise(Y_real)

                with torch.no_grad():
                    tmp_c = G(X_t_c)
                    Yhat_c = tmp_c + sigma * torch.randn_like(tmp_c)

                real_out = D(Y_real)
                fake_out = D(Yhat_c)
                real_score = real_out.view(-1).mean()
                fake_score = fake_out.view(-1).mean()
                loss_D = real_score - fake_score

                D.zero_grad()
                loss_D.backward()
                d_grad = grad_norm_l2(D)
                optimizer_fn(D.parameters(), d_moments, step=update_step, lr=gamma_now,
                             is_discriminator=True, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

            for _ in range(k_generator):
                X_t = sample_noise(Y_real)
                tmp = G(X_t)
                Yhat_t = tmp + sigma * torch.randn_like(tmp)

                c = ((tmp - X_t) ** 2).flatten(1).sum(dim=1)
                d_fake_out = D(Yhat_t).view(-1)
                loss_G = (c - d_fake_out).mean()

                G.zero_grad()
                loss_G.backward()
                g_grad = grad_norm_l2(G)
                optimizer_fn(G.parameters(), g_moments, step=update_step, lr=eta,
                             is_discriminator=False, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

                if G_ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                update_step += 1

            acc['loss_D'] += loss_D.detach()
            acc['loss_G'] += loss_G.detach()
            acc['real_score'] += real_score.detach()
            acc['fake_score'] += fake_score.detach()
            acc['D_grad'] += d_grad
            acc['G_grad'] += g_grad
            acc['D_real_var'] += real_out.var().detach()
            acc['D_fake_var'] += fake_out.var().detach()
            acc['G_out_var'] += Yhat_t.var().detach()
            acc['d_fake_logged'] += d_fake_out.mean().detach()
            acc['c'] += c.mean().detach()
            acc['identity_gap'] += (tmp - X_t).abs().mean().detach()

        avg = {k: (v / num_batches).item() for k, v in acc.items()}

        with torch.no_grad():
            g_drift = sum(
                (p.data - init_params[i]).norm().item()
                for i, p in enumerate(G.parameters())
            )

        G.eval()
        with torch.no_grad():
            fixed_fake = G(fixed_noise).flatten(1)
            diversity = torch.pdist(fixed_fake).mean()
            fixed_fake_eval = G(fixed_eval_noise)
            w2 = compute_w2_squared(fixed_fake_eval, fixed_eval_real)
        G.train()

        _log_epoch_scalars(writer, avg, g_drift, diversity.item(), epoch)
        writer.add_scalar("OT/W2_to_real", w2, epoch)

        if G_ema is not None:
            with torch.no_grad():
                fixed_fake_eval_ema = G_ema(fixed_eval_noise)
                w2_ema = compute_w2_squared(fixed_fake_eval_ema, fixed_eval_real)
            writer.add_scalar("OT/W2_to_real_ema", w2_ema, epoch)

        if epoch % 10 == 0:
            _log_visuals(writer, G, sample_noise, initial_noise_type, Y_real, epoch, device, img_size=img_size)

        if epoch % log_every == 0:
            ckpt_dir = os.path.dirname(f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_ep{epoch}.pth")
            if G_ema is not None:
                torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_ep{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    save_dir = os.path.dirname(f"{model_dir}/{run_name}_G_final.pth")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_final.pth")
    torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_final.pth")
    if G_ema is not None:
        torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_final.pth")
    return G, D


# ============================================================================
# MNIST experiment -- class-conditional (Section 5.4 of the thesis)
# ============================================================================

def train_mnist_cond(G, D, optimizer_fn, train_loader, test_loader,
                      gamma=0.0001, eta=0.0001, sigma=0.001, beta1=0.5, beta2=0.9,
                      num_epochs=200, anchor_power=1, seed=42, run_name='run', device='cpu',
                      k_critic=5, k_generator=1,
                      warmup_steps=80, warmup_k_critic=10,
                      initial_noise_type="uniform",
                      resume_G_path=None, resume_D_path=None,
                      w2_eval_size=256, ema_decay=None,
                      reset_every=10, num_classes=10,
                      collapse_threshold=0.1, collapse_patience=5, collapse_min_epoch=10,
                      img_size=32, log_every=50,
                      log_dir='MNIST_runs', model_dir='models'):
    """Class-conditional primal-dual OT training loop for MNIST.

    Same structure as `train_mnist`, plus: labels flow through `CondGenerator`
    /`ProjCritic`, a per-class diversity metric catches within-class mode
    collapse (the plain diversity metric only catches *across*-class
    collapse), and training stops early if per-class diversity stays below
    `collapse_threshold` for `collapse_patience` consecutive epochs (checked
    only once `epoch >= collapse_min_epoch`). `test_loader` is used for the
    periodic UMAP bottleneck visualization.
    """
    writer = SummaryWriter(f'{log_dir}/{run_name}')
    num_batches = len(train_loader)
    print(f"Steps/epoch: {num_batches}  |  Total steps: {num_batches * num_epochs}")
    if isinstance(gamma, (tuple, list)):
        gi, ga, de = gamma
        print(f"Stepped gamma: {gi} for epochs 1..{de}, then {ga} for epochs {de + 1}..{num_epochs}")

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    update_step = 1
    if resume_G_path is not None:
        G.load_state_dict(torch.load(resume_G_path, map_location=device))
        D.load_state_dict(torch.load(resume_D_path, map_location=device))
        start_epoch = int(re.search(r'_ep(\d+)\.pth', resume_G_path).group(1)) + 1
        update_step = (start_epoch - 1) * num_batches + 1
        print(f"Resumed from epoch {start_epoch - 1}  |  update_step set to {update_step}")

    # ── EMA generator ────────────────────────────────────────────────
    if ema_decay is not None:
        G_ema = deepcopy(G)
        for p in G_ema.parameters():
            p.requires_grad_(False)
        G_ema.eval()
        print(f"EMA enabled: decay={ema_decay}")
    else:
        G_ema = None

    critic_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                shuffle=True, num_workers=train_loader.num_workers)
    critic_iter = iter(critic_loader)

    def get_batch():
        nonlocal critic_iter
        try:
            return next(critic_iter)
        except StopIteration:
            critic_iter = iter(critic_loader)
            return next(critic_iter)

    def make_noise_fn(noise_type):
        if noise_type == "uniform":
            return lambda ref: torch.rand_like(ref)
        elif noise_type == "low_freq":
            return lambda ref: F.interpolate(
                torch.rand(ref.size(0), ref.size(1), 8, 8, device=device),
                size=(ref.size(2), ref.size(3)),
                mode='bilinear', align_corners=False)
        else:
            raise ValueError("incorrect initial_noise_type")
    sample_noise = make_noise_fn(initial_noise_type)

    # Fixed noise + matched labels for diversity tracking.
    torch.manual_seed(seed)
    fixed_noise = torch.rand(64, 1, img_size, img_size, device=device)
    g_cpu = torch.Generator().manual_seed(seed)
    fixed_noise_labels = torch.randint(0, num_classes, (64,), generator=g_cpu).to(device)

    # Fixed eval batches for W² -- labels matched to real eval batch.
    saved_rng = torch.get_rng_state()
    eval_loader = DataLoader(train_loader.dataset, batch_size=w2_eval_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 1))
    fixed_eval_real, fixed_eval_labels = next(iter(eval_loader))
    fixed_eval_real = fixed_eval_real.to(device)
    fixed_eval_labels = fixed_eval_labels.to(device)
    torch.manual_seed(seed + 2)
    fixed_eval_noise = sample_noise(fixed_eval_real).detach()
    torch.set_rng_state(saved_rng)

    init_params = {i: p.data.clone() for i, p in enumerate(G.parameters())}

    d_moments, g_moments = {}, {}
    prev_gamma = None
    collapse_counter = 0

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs"):
        gamma_now = _resolve_gamma(gamma, epoch)
        if prev_gamma is not None and gamma_now != prev_gamma:
            print(f"[epoch {epoch}] gamma stepped: {prev_gamma} -> {gamma_now}")
        prev_gamma = gamma_now
        writer.add_scalar("Schedule/gamma", gamma_now, epoch)

        if epoch == start_epoch:
            _log_hparams(writer, G, D, gamma_now, eta, sigma, k_critic, k_generator,
                         warmup_steps, warmup_k_critic, beta1, optimizer_fn,
                         initial_noise_type, run_name)

        acc = {k: torch.zeros(1, device=device) for k in [
            'loss_D', 'loss_G', 'real_score', 'fake_score',
            'D_grad', 'G_grad', 'D_real_var', 'D_fake_var',
            'G_out_var', 'd_fake_logged', 'c', 'identity_gap',
        ]}

        for Y_t, y_t in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, total=num_batches):
            Y_t = Y_t.to(device)
            y_t = y_t.to(device)
            k_c = warmup_k_critic if update_step < warmup_steps else k_critic

            for _ in range(k_c):
                Y_real, y_real = get_batch()
                Y_real, y_real = Y_real.to(device), y_real.to(device)
                X_t_c = sample_noise(Y_real)

                with torch.no_grad():
                    tmp_c = G(X_t_c, y_real)
                    Yhat_c = tmp_c + sigma * torch.randn_like(tmp_c)

                real_out = D(Y_real, y_real)
                fake_out = D(Yhat_c, y_real)
                real_score = real_out.view(-1).mean()
                fake_score = fake_out.view(-1).mean()
                loss_D = real_score - fake_score

                D.zero_grad()
                loss_D.backward()
                d_grad = grad_norm_l2(D)
                optimizer_fn(D.parameters(), d_moments, step=update_step, lr=gamma_now,
                             is_discriminator=True, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

            for _ in range(k_generator):
                X_t = sample_noise(Y_t)
                tmp = G(X_t, y_t)
                Yhat_t = tmp + sigma * torch.randn_like(tmp)

                c = ((tmp - X_t) ** 2).flatten(1).sum(dim=1)
                d_fake_out = D(Yhat_t, y_t).view(-1)
                loss_G = (c - d_fake_out).mean()

                G.zero_grad()
                loss_G.backward()
                g_grad = grad_norm_l2(G)
                optimizer_fn(G.parameters(), g_moments, step=update_step, lr=eta,
                             is_discriminator=False, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

                if G_ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                update_step += 1

            acc['loss_D'] += loss_D.detach()
            acc['loss_G'] += loss_G.detach()
            acc['real_score'] += real_score.detach()
            acc['fake_score'] += fake_score.detach()
            acc['D_grad'] += d_grad
            acc['G_grad'] += g_grad
            acc['D_real_var'] += real_out.var().detach()
            acc['D_fake_var'] += fake_out.var().detach()
            acc['G_out_var'] += Yhat_t.var().detach()
            acc['d_fake_logged'] += d_fake_out.mean().detach()
            acc['c'] += c.mean().detach()
            acc['identity_gap'] += (tmp - X_t).abs().mean().detach()

        avg = {k: (v / num_batches).item() for k, v in acc.items()}

        with torch.no_grad():
            g_drift = sum((p.data - init_params[i]).norm().item()
                          for i, p in enumerate(G.parameters()))

        G.eval()
        with torch.no_grad():
            fixed_fake = G(fixed_noise, fixed_noise_labels).flatten(1)
            diversity = torch.pdist(fixed_fake).mean()
            fixed_fake_eval = G(fixed_eval_noise, fixed_eval_labels)
            w2 = compute_w2_squared(fixed_fake_eval, fixed_eval_real)

            # Per-class diversity: for each class, fix the label, vary the
            # noise, and measure pairwise distance. This is what actually
            # detects within-class mode collapse (the diversity scalar above
            # conflates across-class and within-class variation).
            n_per = 8
            pc_divs = []
            for cls in range(num_classes):
                y_cls = torch.full((n_per,), cls, device=device, dtype=torch.long)
                ref_pc = torch.empty(n_per, 1, img_size, img_size, device=device)
                x_cls = sample_noise(ref_pc)
                out_pc = G(x_cls, y_cls).flatten(1)
                pc_divs.append(torch.pdist(out_pc).mean().item())
            per_class_diversity = float(np.mean(pc_divs))
        G.train()

        _log_epoch_scalars(writer, avg, g_drift, diversity.item(), epoch)
        writer.add_scalar("OT/W2_to_real", w2, epoch)
        writer.add_scalar("Collapse/per_class_diversity", per_class_diversity, epoch)

        # ── Mode-collapse early stop ─────────────────────────────────
        if collapse_threshold is not None and epoch >= collapse_min_epoch:
            if per_class_diversity < collapse_threshold:
                collapse_counter += 1
            else:
                collapse_counter = 0
            writer.add_scalar("Collapse/early_stop_counter", collapse_counter, epoch)
            if collapse_counter >= collapse_patience:
                print(f"[early stop] per_class_diversity < {collapse_threshold} for "
                      f"{collapse_patience} consecutive epochs — stopping at epoch {epoch}")
                ckpt_dir = os.path.dirname(f"{model_dir}/{run_name}_G_collapse_ep{epoch}.pth")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_collapse_ep{epoch}.pth")
                torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_collapse_ep{epoch}.pth")
                break

        if G_ema is not None:
            with torch.no_grad():
                fixed_fake_eval_ema = G_ema(fixed_eval_noise, fixed_eval_labels)
                w2_ema = compute_w2_squared(fixed_fake_eval_ema, fixed_eval_real)
            writer.add_scalar("OT/W2_to_real_ema", w2_ema, epoch)

        if epoch % 10 == 0:
            fig = plot_conditional_grid(G, sample_noise, Y_t, epoch, device, num_classes=num_classes)
            writer.add_figure("Visuals/Conditional_grid", fig, global_step=epoch)
            plt.close(fig)

            _log_umap(writer, G, test_loader, epoch, device)

        if epoch % log_every == 0:
            ckpt_dir = os.path.dirname(f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_ep{epoch}.pth")
            if G_ema is not None:
                torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_ep{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    save_dir = os.path.dirname(f"{model_dir}/{run_name}_G_final.pth")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_final.pth")
    torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_final.pth")
    if G_ema is not None:
        torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_final.pth")
    return G, D


# ============================================================================
# MNIST experiment -- conditional WGAN-GP baseline (Section 5.5 of the thesis)
# ============================================================================

def cond_gradient_penalty(D, real, fake, y, device):
    """lambda * E[(||grad_x D(x_hat, y)||_2 - 1)^2], x_hat = alpha*real + (1-alpha)*fake.

    Same as the unconditional gradient penalty,
    with one change: the critic also sees the label y. Only the IMAGES are
    interpolated -- labels are discrete (you can't have 0.3*"7" + 0.7*"3"),
    so each x_hat keeps the label of the real/fake pair it was built from.
    The penalty therefore enforces 1-Lipschitzness of x -> D(x, y)
    separately within each class slice, exactly the constraint the
    conditional Wasserstein objective needs.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = alpha * real + (1.0 - alpha) * fake
    x_hat.requires_grad_(True)
    d_x_hat = D(x_hat, y).view(-1)
    grads = torch.autograd.grad(
        outputs=d_x_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_x_hat),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def train_mnist_cond_wgan_gp(G, D, train_loader, num_epochs=200, n_critic=5,
                              lambda_gp=10.0, lr=1e-4, beta1=0.0, beta2=0.9,
                              img_size=32, seed=42, run_name="cwgan_gp",
                              device='cpu', num_classes=10,
                              w2_eval_size=256, log_every_visuals=10, ckpt_every=50,
                              log_dir='MNIST_runs', model_dir='models'):
    """Conditional WGAN-GP baseline: `CondGenerator` + `ProjCritic` trained
    with the standard WGAN-GP objective (plain Adam, no sigma-noise, no
    anchoring, no warmup) instead of the primal-dual OT objective. Used in
    the thesis to isolate how much of the conditional model's performance
    comes from the objective/optimizer vs. the conditioning architecture.
    """
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    num_batches = len(train_loader)
    print(f"Steps/epoch: {num_batches}  |  Total steps: {num_batches * num_epochs}")
    print(f"cWGAN-GP: n_critic={n_critic}, lambda_gp={lambda_gp}, "
          f"Adam(lr={lr}, betas=({beta1}, {beta2}))")

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    critic_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                shuffle=True, num_workers=train_loader.num_workers)
    critic_iter = iter(critic_loader)

    def get_batch():
        nonlocal critic_iter
        try:
            return next(critic_iter)
        except StopIteration:
            critic_iter = iter(critic_loader)
            return next(critic_iter)

    def sample_noise(ref):
        """Matches the B/C-series `low_freq` noise: bilinear upsample from 8x8 random."""
        return F.interpolate(
            torch.rand(ref.size(0), ref.size(1), 8, 8, device=device),
            size=(ref.size(2), ref.size(3)),
            mode="bilinear", align_corners=False,
        )

    # Fixed noise + matched labels for diversity tracking (same seeds as train_mnist_cond).
    torch.manual_seed(seed)
    fixed_noise = torch.rand(64, 1, img_size, img_size, device=device)
    g_cpu = torch.Generator().manual_seed(seed)
    fixed_noise_labels = torch.randint(0, num_classes, (64,), generator=g_cpu).to(device)

    # Fixed eval batches for W² -- labels matched to real eval batch.
    saved_rng = torch.get_rng_state()
    eval_loader = DataLoader(train_loader.dataset, batch_size=w2_eval_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 1))
    fixed_eval_real, fixed_eval_labels = next(iter(eval_loader))
    fixed_eval_real = fixed_eval_real.to(device)
    fixed_eval_labels = fixed_eval_labels.to(device)
    torch.manual_seed(seed + 2)
    fixed_eval_noise = sample_noise(fixed_eval_real).detach()
    torch.set_rng_state(saved_rng)

    update_step = 1
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        acc = {k: 0.0 for k in [
            "loss_D", "loss_G", "gp", "real_score", "fake_score",
            "D_grad", "G_grad", "G_out_var",
        ]}

        for Y_t, y_t in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, total=num_batches):
            Y_t = Y_t.to(device)
            y_t = y_t.to(device)

            # ── Critic: n_critic updates ────────────────────────────
            for _ in range(n_critic):
                Y_real, y_real = get_batch()
                Y_real, y_real = Y_real.to(device), y_real.to(device)
                X = sample_noise(Y_real)
                with torch.no_grad():
                    Yhat = G(X, y_real)  # fake conditioned on REAL labels

                real_out = D(Y_real, y_real).view(-1)
                fake_out = D(Yhat, y_real).view(-1)
                gp = cond_gradient_penalty(D, Y_real, Yhat, y_real, device)
                loss_D = fake_out.mean() - real_out.mean() + lambda_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                d_grad = grad_norm_l2(D)
                opt_D.step()

            # ── Generator: 1 update (labels from the main loader batch) ──
            X = sample_noise(Y_t)
            Yhat = G(X, y_t)
            d_fake = D(Yhat, y_t).view(-1)
            loss_G = -d_fake.mean()  # pure WGAN: no transport cost

            opt_G.zero_grad()
            loss_G.backward()
            g_grad = grad_norm_l2(G)
            opt_G.step()

            update_step += 1

            acc["loss_D"] += loss_D.detach().item()
            acc["loss_G"] += loss_G.detach().item()
            acc["gp"] += gp.detach().item()
            acc["real_score"] += real_out.mean().detach().item()
            acc["fake_score"] += fake_out.mean().detach().item()
            acc["D_grad"] += d_grad
            acc["G_grad"] += g_grad
            acc["G_out_var"] += Yhat.var().detach().item()

        avg = {k: v / num_batches for k, v in acc.items()}

        # ── Eval: W², diversity, per-class diversity (same as train_mnist_cond) ──
        G.eval()
        with torch.no_grad():
            fixed_fake_flat = G(fixed_noise, fixed_noise_labels).flatten(1)
            diversity = torch.pdist(fixed_fake_flat).mean()
            fixed_fake_eval = G(fixed_eval_noise, fixed_eval_labels)
            w2 = compute_w2_squared(fixed_fake_eval, fixed_eval_real)

            n_per = 8
            pc_divs = []
            for cls in range(num_classes):
                y_cls = torch.full((n_per,), cls, device=device, dtype=torch.long)
                ref_pc = torch.empty(n_per, 1, img_size, img_size, device=device)
                x_cls = sample_noise(ref_pc)
                out_pc = G(x_cls, y_cls).flatten(1)
                pc_divs.append(torch.pdist(out_pc).mean().item())
            per_class_diversity = float(np.mean(pc_divs))
        G.train()

        for k, v in avg.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        writer.add_scalar("OT/W2_to_real", w2, epoch)
        writer.add_scalar("Diversity/fixed_noise_pdist", diversity.item(), epoch)
        writer.add_scalar("Collapse/per_class_diversity", per_class_diversity, epoch)

        if epoch % log_every_visuals == 0:
            fig = plot_conditional_grid(G, sample_noise, Y_t, epoch, device, num_classes=num_classes)
            writer.add_figure("Visuals/Conditional_grid", fig, global_step=epoch)
            plt.close(fig)

        if epoch % ckpt_every == 0:
            ckpt_dir = os.path.dirname(f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_ep{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    save_dir = os.path.dirname(f"{model_dir}/{run_name}_G_final.pth")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_final.pth")
    torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_final.pth")
    return G, D


# ============================================================================
# CartoonSet experiment -- CLIP fine-tuning (Section 6.3 of the thesis)
#
# This is a separate contrastive fine-tuning stage that runs *before* the
# OT-GAN training loops above: its output (a fine-tuned CLIP checkpoint +
# precomputed image embeddings) becomes the conditioning signal the
# CartoonSet generator/critic are trained on.
# ============================================================================

def clip_loss(image_features, text_features, logit_scale):
    """Symmetric CLIP contrastive loss on the [B x B] similarity matrix.
    Diagonal entries are the correct (image, text) pairs."""
    img = F.normalize(image_features, dim=-1)
    txt = F.normalize(text_features, dim=-1)

    logits_img = logit_scale.exp() * img @ txt.T  # [B, B]
    logits_txt = logits_img.T

    labels = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits_img, labels) +
            F.cross_entropy(logits_txt, labels)) / 2


def setup_trainable_params(model, n_visual_blocks=2, n_text_blocks=2):
    """Freeze everything, then selectively unfreeze the last few transformer
    blocks of each tower (plus projections and the temperature).

    Keeping 95%+ of the model frozen prevents catastrophic forgetting of
    CLIP's general representations while still letting it specialize to the
    cartoon domain (Section 6.3 of the thesis).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    for block in model.visual.transformer.resblocks[-n_visual_blocks:]:
        for p in block.parameters():
            p.requires_grad_(True)

    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        model.visual.proj.requires_grad_(True)

    for block in model.transformer.resblocks[-n_text_blocks:]:
        for p in block.parameters():
            p.requires_grad_(True)

    if model.text_projection is not None:
        model.text_projection.requires_grad_(True)

    model.logit_scale.requires_grad_(True)  # temperature -- always trained

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")

    return [p for p in model.parameters() if p.requires_grad]


def finetune_clip(df, resume_path=None, save_path='clip_cartoon_finetuned.pt',
                   model_name='ViT-B-32', pretrained='openai',
                   batch_size=128, num_epochs=5, lr=5e-6,
                   n_visual_blocks=2, n_text_blocks=2,
                   num_workers=2, device='cuda', seed=42):
    """Fine-tunes the last few layers of CLIP on cartoon (image, caption)
    pairs, with linear warmup + cosine decay.

    `df` needs `img_path` and `clip_text_description` columns (see
    `captions.generate_clip_caption`).

    Returns (model, tokenizer, preprocess) -- `model` is in eval mode.
    """
    import open_clip

    torch.manual_seed(seed)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Resumed from {resume_path}")

    trainable_params = setup_trainable_params(model, n_visual_blocks, n_text_blocks)

    dataset = CartoonCLIPDataset(df, tokenizer, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    print(f"{len(dataset):,} samples  |  {steps_per_epoch} steps/epoch  |  "
          f"{total_steps} total steps  |  {warmup_steps} warmup steps")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for imgs, texts in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs = imgs.to(device)
            texts = texts.to(device)

            image_features = model.encode_image(imgs)
            text_features = model.encode_text(texts)
            loss = clip_loss(image_features, text_features, model.logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))  # cap temperature at 100

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch}  |  loss: {avg_loss:.4f}  |  "
              f"logit_scale: {model.logit_scale.exp().item():.2f}  |  "
              f"lr: {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved -> {save_path}")

    model.eval()
    return model, tokenizer, preprocess


# ============================================================================
# CartoonSet experiment -- CLIP-conditioned OT-GAN training (Section 6.4-6.5
# of the thesis). Runs after CLIP fine-tuning above: the generator/critic
# condition on the *precomputed* embeddings from `precompute_embeddings`,
# never running CLIP itself during GAN training.
# ============================================================================

def apply_cfg_dropout_clip(clip_emb, null_token, p_uncond=0.20):
    """replaces each [512] embedding row
    with the generator's learned null token with probability `p_uncond`, so
    the model learns both conditional and unconditional generation in the
    same training run."""
    mask = torch.rand(clip_emb.size(0), device=clip_emb.device) < p_uncond
    null = null_token.unsqueeze(0).expand_as(clip_emb)
    return torch.where(mask.unsqueeze(1), null, clip_emb)


def get_fixed_text_embs(train_df, clip_model, tokenizer, device):
    """Picks one real caption per target hair color from `train_df` and
    encodes it with the fine-tuned CLIP text encoder -- used as a fixed,
    reproducible set of visualization queries logged every 10 epochs.

    Returns: (fixed_text_embs [5, 512] on device, viz_labels: list[str]).
    """
    targets = {'blonde': 0, 'ginger': 2, 'dark brown': 6, 'black': 7, 'grey': 8}
    clip_model.eval()
    embs, labels = [], []

    for color_name, color_id in targets.items():
        subset = train_df[train_df['hair_color'] == color_id]
        if len(subset) == 0:
            raise ValueError(f"No training samples with hair_color={color_id}")
        text = subset.iloc[0]['clip_text_description']

        with torch.no_grad():
            tok = tokenizer([text]).to(device)
            emb = F.normalize(clip_model.encode_text(tok), dim=-1)

        embs.append(emb)
        labels.append(color_name)
        print(f"[{color_name:>10}]  {text[:70]}...")

    return torch.cat(embs, dim=0), labels


def _log_visuals_clip(writer, G, fixed_noise, fixed_text_embs, viz_labels, epoch, device):
    """Row 1: 5 text-conditioned samples (one per hair-color query). Row 2:
    the same noise through the null token (unconditional). `G` must be
    wrapped in `nn.DataParallel` (accesses `G.module.null_token`) -- see
    `run_cartoon_experiment`, which always wraps it, even on a single GPU."""
    G.eval()
    n = fixed_noise.size(0)

    with torch.no_grad():
        row1 = G(fixed_noise, fixed_text_embs).cpu().clamp(0, 1)
        null = G.module.null_token.detach().unsqueeze(0).expand(n, -1)
        row2 = G(fixed_noise, null).cpu().clamp(0, 1)

    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
    for i in range(n):
        axes[0, i].imshow(row1[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(viz_labels[i], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(row2[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title('null', fontsize=8)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('conditioned', fontsize=8)
    axes[1, 0].set_ylabel('null token', fontsize=8)
    fig.suptitle(f'Epoch {epoch}', fontsize=10)
    fig.tight_layout()
    writer.add_figure("Visuals/CLIP_Conditional", fig, global_step=epoch)
    plt.close(fig)
    G.train()


def train_clip_cond(G, D, optimizer_fn, train_loader,
                     fixed_text_embs, viz_labels,
                     p_uncond=0.20, gamma=0.0001, eta=0.0001, sigma=0.001,
                     beta1=0.5, beta2=0.9, num_epochs=200, anchor_power=1,
                     seed=42, run_name='run', device='cuda',
                     k_critic=5, k_generator=1,
                     warmup_steps=80, warmup_k_critic=10,
                     resume_G_path=None, resume_D_path=None,
                     w2_eval_size=256, ema_decay=None, reset_every=10,
                     use_amp=False, lean_data_path=False,
                     img_size=64, channels=3,
                     log_dir='Cartoon_runs', model_dir='models'):
    """Primal-dual OT training loop for CLIP-conditioned CartoonSet
    generation. Structurally the same primal-dual game as `train_mnist_cond`
    (critic warmup, anchor-reset optimizers, EMA generator), with three
    CartoonSet-specific differences: "CFG" dropout on the conditioning
    embedding, an 8x8->img_size bilinear-upsampled noise source (matching
    the low-frequency-noise fix from the MNIST experiments), and optional
    AMP (`use_amp`) since these images are much larger than MNIST's.

    `G`/`D` must already be wrapped in `nn.DataParallel` (this function
    accesses `G.module.null_token`) -- `run_cartoon_experiment` handles this
    for you.
    """
    from contextlib import nullcontext

    writer = SummaryWriter(f'{log_dir}/{run_name}')
    num_batches = len(train_loader)
    print(f"Steps/epoch: {num_batches}  |  Total: {num_batches * num_epochs}")

    amp_ctx = ((lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16))
               if use_amp else nullcontext)

    start_epoch = 1
    update_step = 1
    if resume_G_path is not None:
        G.load_state_dict(torch.load(resume_G_path, map_location=device))
        D.load_state_dict(torch.load(resume_D_path, map_location=device))
        m = re.search(r'_ep(\d+)\.pth', resume_G_path)
        start_epoch = (int(m.group(1)) + 1) if m else 1
        update_step = (start_epoch - 1) * num_batches + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    if ema_decay is not None:
        G_ema = deepcopy(G)
        for p in G_ema.parameters():
            p.requires_grad_(False)
        G_ema.eval()
    else:
        G_ema = None

    critic_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                shuffle=True, num_workers=train_loader.num_workers, pin_memory=True)
    critic_iter = iter(critic_loader)

    def get_batch():
        nonlocal critic_iter
        try:
            return next(critic_iter)
        except StopIteration:
            critic_iter = iter(critic_loader)
            return next(critic_iter)

    def sample_noise(ref):
        noise_small = torch.rand(ref.size(0), channels, 8, 8, device=ref.device)
        return F.interpolate(noise_small, size=(img_size, img_size), mode='bilinear', align_corners=False)

    torch.manual_seed(seed)
    small = torch.rand(5, channels, 8, 8, device=device)
    fixed_noise = F.interpolate(small, size=(img_size, img_size), mode='bilinear', align_corners=False)

    saved_rng = torch.get_rng_state()
    eval_imgs, eval_embs = next(iter(DataLoader(
        train_loader.dataset, batch_size=w2_eval_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1))))
    fixed_eval_real = eval_imgs.to(device)
    fixed_eval_embs = eval_embs.to(device)
    torch.manual_seed(seed + 2)
    small_eval = torch.rand(w2_eval_size, channels, 8, 8, device=device)
    fixed_eval_noise = F.interpolate(small_eval, size=(img_size, img_size), mode='bilinear', align_corners=False)
    torch.set_rng_state(saved_rng)

    init_params = {i: p.data.clone() for i, p in enumerate(G.parameters())}
    d_moments, g_moments = {}, {}

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs"):
        acc = {k: torch.zeros(1, device=device) for k in [
            'loss_D', 'loss_G', 'real_score', 'fake_score',
            'D_grad', 'G_grad', 'D_real_var', 'D_fake_var',
            'G_out_var', 'd_fake_logged', 'c', 'identity_gap',
        ]}

        # lean_data_path=False -> iterate train_loader exactly as the original
        # (preserves shuffle RNG); the yielded batch is intentionally unused.
        epoch_iter = range(num_batches) if lean_data_path else train_loader
        for _batch in tqdm(epoch_iter, desc=f"Epoch {epoch}", leave=False, total=num_batches):
            k_c = warmup_k_critic if update_step < warmup_steps else k_critic

            for _ in range(k_c):
                Y_real, emb_real = get_batch()
                Y_real = Y_real.to(device)
                emb_real = emb_real.to(device)
                X_t_c = sample_noise(Y_real)
                emb_dropped = apply_cfg_dropout_clip(emb_real, G.module.null_token, p_uncond)

                with torch.no_grad():
                    with amp_ctx():
                        tmp_c = G(X_t_c, emb_dropped)
                    if use_amp:
                        tmp_c = tmp_c.float()
                    Yhat_c = tmp_c + sigma * torch.randn_like(tmp_c)

                with amp_ctx():
                    real_out = D(Y_real, emb_dropped)
                    fake_out = D(Yhat_c, emb_dropped)
                if use_amp:
                    real_out = real_out.float()
                    fake_out = fake_out.float()
                real_score = real_out.view(-1).mean()
                fake_score = fake_out.view(-1).mean()
                loss_D = real_score - fake_score

                D.zero_grad()
                loss_D.backward()
                d_grad = grad_norm_l2_tensor(D)
                optimizer_fn(D.parameters(), d_moments, step=update_step, lr=gamma,
                             is_discriminator=True, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

            for _ in range(k_generator):
                X_t = sample_noise(Y_real)
                emb_dropped = apply_cfg_dropout_clip(emb_real, G.module.null_token, p_uncond)

                with amp_ctx():
                    tmp = G(X_t, emb_dropped)
                if use_amp:
                    tmp = tmp.float()
                Yhat_t = tmp + sigma * torch.randn_like(tmp)

                c = ((tmp - X_t) ** 2).flatten(1).sum(dim=1)
                with amp_ctx():
                    d_fake_out = D(Yhat_t, emb_dropped).view(-1)
                if use_amp:
                    d_fake_out = d_fake_out.float()
                loss_G = (c - d_fake_out).mean()

                G.zero_grad()
                loss_G.backward()
                g_grad = grad_norm_l2_tensor(G)
                optimizer_fn(G.parameters(), g_moments, step=update_step, lr=eta,
                             is_discriminator=False, beta1=beta1, beta2=beta2, power=anchor_power,
                             reset_every=reset_every, steps_per_epoch=num_batches)

                if G_ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                update_step += 1

            acc['loss_D'] += loss_D.detach()
            acc['loss_G'] += loss_G.detach()
            acc['real_score'] += real_score.detach()
            acc['fake_score'] += fake_score.detach()
            acc['D_grad'] += d_grad
            acc['G_grad'] += g_grad
            acc['D_real_var'] += real_out.var().detach()
            acc['D_fake_var'] += fake_out.var().detach()
            acc['G_out_var'] += Yhat_t.var().detach()
            acc['d_fake_logged'] += d_fake_out.mean().detach()
            acc['c'] += c.mean().detach()
            acc['identity_gap'] += (tmp - X_t).abs().mean().detach()

        avg = {k: (v / num_batches).item() for k, v in acc.items()}

        with torch.no_grad():
            g_drift = sum((p.data - init_params[i]).norm().item() for i, p in enumerate(G.parameters()))

        G.eval()
        with torch.no_grad():
            fixed_fake = G(fixed_noise, fixed_text_embs).flatten(1)
            diversity = torch.pdist(fixed_fake).mean()
            fixed_fake_eval = G(fixed_eval_noise, fixed_eval_embs)
            w2 = compute_w2_squared_sliced(fixed_fake_eval, fixed_eval_real)
        G.train()

        _log_epoch_scalars(writer, avg, g_drift, diversity.item(), epoch)
        writer.add_scalar("OT/W2_to_real", w2, epoch)

        if G_ema is not None:
            G_ema.eval()
            with torch.no_grad():
                w2_ema = compute_w2_squared_sliced(G_ema(fixed_eval_noise, fixed_eval_embs), fixed_eval_real)
            writer.add_scalar("OT/W2_to_real_ema", w2_ema, epoch)

        if epoch % 10 == 0:
            _log_visuals_clip(writer, G, fixed_noise, fixed_text_embs, viz_labels, epoch, device)
            ckpt_dir = os.path.dirname(f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_ep{epoch}.pth")
            torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_ep{epoch}.pth")
            if G_ema is not None:
                torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_ep{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    save_dir = os.path.dirname(f"{model_dir}/{run_name}_G_final.pth")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G.state_dict(), f"{model_dir}/{run_name}_G_final.pth")
    torch.save(D.state_dict(), f"{model_dir}/{run_name}_D_final.pth")
    if G_ema is not None:
        torch.save(G_ema.state_dict(), f"{model_dir}/{run_name}_Gema_final.pth")
    return G, D


# --- Experiment runner: hyperparameters, run naming, resuming, launching ---

_CARTOON_BASELINE = dict(gamma=1e-5, eta=1e-4, sigma=0.01, beta1=0.5, k_critic=2,
                          k_generator=1, p_uncond=0.20, reset_every=50, num_epochs=200,
                          seed=42, train_size=50_000, batch_size=64, features_g=64,
                          opt="aadam", arch="adagn")
_CARTOON_ARCH_CODE = {"adagn": "adagn", "adagn_attention": "attn", "fully_injected": "fullinj"}
_CARTOON_ABBR = dict(gamma="g", eta="eta", sigma="s", reset_every="r", k_critic="kc",
                      k_generator="kg", num_epochs="ep", train_size="n", batch_size="bs",
                      features_g="fg", beta1="b1", p_uncond="pu", seed="seed", opt="", arch="")

_CARTOON_OPTIMIZERS = {
    "adam": adam_update,
    "optimistic_adam": optimistic_adam_update,
    "anchored_adam": anchored_adam_update,
    "anchored_optimistic_adam": anchored_optimistic_adam_update,
}
_CARTOON_OPT_CODE = {"adam": "adam", "optimistic_adam": "oadam",
                      "anchored_adam": "aadam", "anchored_optimistic_adam": "aoadam"}


def _wrap_optimizer(fn):
    """Filters kwargs down to what `fn` actually accepts, so the same call
    site can pass anchoring-specific kwargs (power/reset_every/...) to plain
    `adam_update`/`optimistic_adam_update` without a TypeError."""
    import inspect
    p = inspect.signature(fn).parameters
    if any(x.kind == x.VAR_KEYWORD for x in p.values()):
        return fn
    accepted = set(p)

    def wrapped(*args, **kw):
        return fn(*args, **{k: v for k, v in kw.items() if k in accepted})
    return wrapped


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def build_cartoon_run_name(exp_id, note="", **hparams):
    """Builds a run name that only encodes the hyperparameters that differ
    from `_CARTOON_BASELINE` -- keeps TensorBoard run names short and
    comparable at a glance."""
    diffs = {k: v for k, v in hparams.items() if k in _CARTOON_BASELINE and v != _CARTOON_BASELINE[k]}
    parts = [f"{_CARTOON_ABBR.get(k, k)}{_fmt(v)}" for k, v in sorted(diffs.items())]
    tag = "_".join(parts) if parts else "baseline"
    name = f"{exp_id}/{tag}"
    if note:
        slug = re.sub(r'[^a-z0-9]+', '-', note.lower()).strip('-')
        name = f"{name}__{slug}"
    return name


def _find_latest_cartoon_ckpt(run_name, model_dir='models'):
    import glob
    gs = glob.glob(f"{model_dir}/{run_name}_G_ep*.pth")
    if not gs:
        return None, None
    latest = max(gs, key=lambda p: int(re.search(r'_ep(\d+)\.pth', p).group(1)))
    return latest, latest.replace("_G_ep", "_D_ep")


def run_cartoon_experiment(exp_id, note="", *, device = "cuda",
                            metadata_csv, embeddings_path, images_dir, clip_weights_path,
                            arch="adagn",
                            train_size=50_000, test_size=1_000, batch_size=64, num_workers=0,
                            features_g=64, features_d=64,
                            gamma=1e-5, eta=1e-4, sigma=0.01, beta1=0.5, beta2=0.9,
                            k_critic=2, k_generator=1, p_uncond=0.20, reset_every=50,
                            num_epochs=200, seed=42, anchor_power=1,
                            warmup_epochs=10, warmup_k_critic=5, w2_eval_size=256, ema_decay=None,
                            optimizer="anchored_adam",
                            allow_tf32=False, cudnn_benchmark=False,
                            use_amp=False, lean_data_path=False,
                            img_size=64, channels=3, clip_dim=512, ch_cap=512,
                            log_dir='Cartoon_runs', model_dir='models',
                            smoke=False, dry_run=False, resume="none"):
    """Experiment launcher for CLIP-conditioned CartoonSet training: builds
    the generator/critic for the requested `arch`, wraps them in
    `nn.DataParallel` (required by `train_clip_cond`), loads the fixed
    visualization queries, and dispatches to `train_clip_cond`.

    Requires a CUDA device (uses `nn.DataParallel` and hardcodes
    `device='cuda'`, matching the original -- this pipeline was never run
    on CPU). `dry_run=True` builds the models and does one shape/backward
    check without loading data or training, useful for validating a config
    before committing a full run.
    """
    from models import ClipCondGeneratorAdaGN, GeneratorAdaGNAttention, ClipCondGeneratorFullyInjected, ClipProjCritic

    generators = {
        "adagn": ClipCondGeneratorAdaGN,
        "adagn_attention": GeneratorAdaGNAttention,
        "fully_injected": ClipCondGeneratorFullyInjected,
    }

    device = device

    if arch not in generators:
        raise ValueError(f"arch must be one of {list(generators)}; got {arch!r}")
    if optimizer not in _CARTOON_OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {list(_CARTOON_OPTIMIZERS)}; got {optimizer!r}")
    optimizer_fn = _wrap_optimizer(_CARTOON_OPTIMIZERS[optimizer])
    Generator = generators[arch]

    if smoke:
        train_size, num_epochs = min(train_size, 512), 2
        print("SMOKE TEST -> train_size=512, num_epochs=2")

    run_name = build_cartoon_run_name(
        exp_id, note, gamma=gamma, eta=eta, sigma=sigma, beta1=beta1,
        k_critic=k_critic, k_generator=k_generator, p_uncond=p_uncond,
        reset_every=reset_every, num_epochs=num_epochs, seed=seed,
        train_size=train_size, batch_size=batch_size, features_g=features_g,
        opt=_CARTOON_OPT_CODE[optimizer], arch=_CARTOON_ARCH_CODE[arch])
    print(f"RUN: {run_name}")
    print(f"arch: {arch}  |  speed: tf32={allow_tf32} cudnn_bench={cudnn_benchmark} "
          f"amp={use_amp} lean={lean_data_path}")

    if dry_run:
        print("DRY RUN - no data loaded, no training.")
        print(f"  arch      : {arch}  (run-name code: {_CARTOON_ARCH_CODE[arch]})")
        print(f"  optimizer : {optimizer}  (run-name code: {_CARTOON_OPT_CODE[optimizer]})")
        print(f"  data      : train_size={train_size}, batch_size={batch_size}, test_size={test_size}")
        print(f"  hparams   : gamma={gamma}, eta={eta}, sigma={sigma}, beta1={beta1}, "
              f"k_critic={k_critic}, reset_every={reset_every}, num_epochs={num_epochs}, seed={seed}")
        rng = torch.get_rng_state()
        crng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        G = Generator(channels_img=3, features_g=features_g, clip_dim=clip_dim, ch_cap=ch_cap).to(device)
        D = ClipProjCritic(channels_img=3, features_d=features_d, clip_dim=clip_dim, ch_cap=ch_cap,
                            img_size=img_size).to(device)
        print(f"  G params  : {sum(p.numel() for p in G.parameters()):,}")
        print(f"  D params  : {sum(p.numel() for p in D.parameters()):,}")
        n_ = torch.rand(4, 3, img_size, img_size, device=device)
        e_ = torch.randn(4, clip_dim, device=device)
        o_ = G(n_, e_)
        s_ = D(o_, e_)
        s_.mean().backward()
        print(f"  shapes    : {tuple(o_.shape)} {tuple(s_.shape)} — backward OK")
        torch.set_rng_state(rng)
        if crng is not None:
            torch.cuda.set_rng_state_all(crng)
        print("  -> looks right? set dry_run=False to launch.")
        return None

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader, train_df, test_df, embeddings = get_clip_loaders(
        metadata_csv=metadata_csv, embeddings_path=embeddings_path, images_dir=images_dir,
        img_size=img_size, batch_size=batch_size,
        train_size=train_size, test_size=test_size, num_workers=num_workers)

    G = Generator(channels_img=3, features_g=features_g, clip_dim=clip_dim, ch_cap=ch_cap).to(device)
    D = ClipProjCritic(channels_img=3, features_d=features_d, clip_dim=clip_dim, ch_cap=ch_cap,
                        img_size=img_size).to(device)
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}  |  "
          f"D params: {sum(p.numel() for p in D.parameters()):,}")

    # Null token = mean embedding of the training split, set BEFORE DataParallel wraps G.
    G.null_token.data.copy_(torch.tensor(
        embeddings[train_df['emb_idx'].values].mean(axis=0), dtype=torch.float32).to(device))

    print(f"Using {torch.cuda.device_count()} GPU(s)")
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model.load_state_dict(torch.load(clip_weights_path, map_location=device))
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    fixed_text_embs, viz_labels = get_fixed_text_embs(train_df, clip_model, tokenizer, device)
    del clip_model
    torch.cuda.empty_cache()

    resume_G_path = resume_D_path = None
    if resume == "auto":
        resume_G_path, resume_D_path = _find_latest_cartoon_ckpt(run_name, model_dir)
        print(f"auto-resume from {resume_G_path}" if resume_G_path
              else "auto-resume: no checkpoint found, starting fresh")
    elif resume not in ("none", None):
        resume_G_path, resume_D_path = f"{resume}_G.pth", f"{resume}_D.pth"

    return train_clip_cond(
        G, D, optimizer_fn, train_loader,
        fixed_text_embs=fixed_text_embs, viz_labels=viz_labels,
        p_uncond=p_uncond, gamma=gamma, eta=eta, sigma=sigma,
        beta1=beta1, beta2=beta2, num_epochs=num_epochs, anchor_power=anchor_power,
        seed=seed, run_name=run_name, device=device,
        k_critic=k_critic, k_generator=k_generator,
        warmup_steps=len(train_loader) * warmup_epochs, warmup_k_critic=warmup_k_critic,
        resume_G_path=resume_G_path, resume_D_path=resume_D_path,
        w2_eval_size=w2_eval_size, ema_decay=ema_decay, reset_every=reset_every,
        use_amp=use_amp, lean_data_path=lean_data_path,
        img_size=img_size, channels=channels, log_dir=log_dir, model_dir=model_dir)


def sanity_check_cartoon(arch="adagn", optimizer="anchored_adam", features_g=64, features_d=64,
                          img_size=64, clip_dim=512, device='cuda'):
    """Quick param-count / shape / backward-pass check before committing to
    a full `run_cartoon_experiment` launch."""
    from models import ClipCondGeneratorAdaGN, GeneratorAdaGNAttention, ClipCondGeneratorFullyInjected, ClipProjCritic

    generators = {
        "adagn": ClipCondGeneratorAdaGN,
        "adagn_attention": GeneratorAdaGNAttention,
        "fully_injected": ClipCondGeneratorFullyInjected,
    }
    assert arch in generators, f"unknown arch {arch!r}; choose from {list(generators)}"
    assert optimizer in _CARTOON_OPTIMIZERS, f"unknown optimizer {optimizer!r}"

    G = generators[arch](channels_img=3, features_g=features_g, clip_dim=clip_dim).to(device)
    D = ClipProjCritic(channels_img=3, features_d=features_d, clip_dim=clip_dim, img_size=img_size).to(device)
    print(f"arch:      {arch}")
    print(f"G params:  {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params:  {sum(p.numel() for p in D.parameters()):,}")
    print(f"Optimizer: {optimizer}  (run-name code: {_CARTOON_OPT_CODE[optimizer]})")

    noise = torch.rand(4, 3, img_size, img_size, device=device)
    clip_emb = torch.randn(4, clip_dim, device=device)
    fake = G(noise, clip_emb)
    score = D(fake, clip_emb)
    print("shapes:   ", tuple(fake.shape), tuple(score.shape))

    score.mean().backward()
    print("Backward pass OK")
