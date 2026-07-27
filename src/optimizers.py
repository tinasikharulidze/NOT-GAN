"""
Update rules for the primal-dual OT training loop, plus LR schedule helpers.

Each `*_update` function mutates a list of `params` in place using their
`.grad` (already populated by `loss.backward()`), and clears the grad
afterwards. `moments` is a plain dict the caller keeps alive across steps —
it holds per-parameter Adam moments (and, for the anchored variants, the
anchor point each parameter is pulled back towards).

Sign convention: `is_discriminator=True` does gradient ASCENT (the critic
maximizes its loss), everything else does gradient DESCENT.
"""

import torch


def SGD_update(params, moments, step, lr, is_discriminator=False, **kwargs):
    """Plain SGD. `moments`/`step` are unused but kept so every optimizer
    shares one call signature and can be swapped in `train()` without
    special-casing."""
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            if is_discriminator:
                p.data += lr * p.grad
            else:
                p.data -= lr * p.grad
            p.grad.zero_()


def adam_update(params, moments, step, lr, beta1=0.5, beta2=0.999, eps=1e-8,
                 is_discriminator=False, **kwargs):
    """Standard Adam"""
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            if i not in moments:
                moments[i] = {'m': torch.zeros_like(p.data),
                              'v': torch.zeros_like(p.data)}
            m_prev, v_prev = moments[i]['m'], moments[i]['v']
            g = p.grad
            m_new = beta1 * m_prev + (1 - beta1) * g
            v_new = beta2 * v_prev + (1 - beta2) * (g * g)
            moments[i]['m'], moments[i]['v'] = m_new, v_new

            m_hat = m_new / (1 - beta1 ** step)
            v_hat = v_new / (1 - beta2 ** step)
            update = m_hat / (torch.sqrt(v_hat) + eps)

            if is_discriminator:
                p.data += lr * update
            else:
                p.data -= lr * update
            p.grad.zero_()


def optimistic_adam_update(params, moments, step, lr, beta1=0.5, beta2=0.999, eps=1e-8,
                            is_discriminator=False, **kwargs):
    """Adam with the optimistic extrapolation step `2*u_t - u_{t-1}`.

    Reusing the previous update anticipates where the next gradient will
    point, which damps the oscillations typical of saddle-point games.
    """
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            if i not in moments:
                moments[i] = {'m': torch.zeros_like(p.data),
                              'v': torch.zeros_like(p.data),
                              'prev_update': torch.zeros_like(p.data)}
            m_prev = moments[i]['m']
            v_prev = moments[i]['v']
            u_prev = moments[i]['prev_update']
            g = p.grad
            m_new = beta1 * m_prev + (1 - beta1) * g
            v_new = beta2 * v_prev + (1 - beta2) * (g * g)
            moments[i]['m'], moments[i]['v'] = m_new, v_new

            m_hat = m_new / (1 - beta1 ** step)
            v_hat = v_new / (1 - beta2 ** step)
            u_current = m_hat / (torch.sqrt(v_hat) + eps)
            optimistic_u = 2 * u_current - u_prev
            moments[i]['prev_update'] = u_current

            if is_discriminator:
                p.data = p.data + lr * optimistic_u
            else:
                p.data = p.data - lr * optimistic_u
            p.grad.zero_()


def anchored_adam_update(params, moments, step, lr, beta1=0.5, beta2=0.999, eps=1e-8,
                          power=1, reset_every=None, steps_per_epoch=None,
                          is_discriminator=False, **kwargs):
    """Adam with Halpern-style anchoring toward an earlier "anchor" point.

    After each Adam step, parameters are pulled back towards the anchor:
        p_{t+1} <- beta_t * anchor + (1 - beta_t) * p_{t+1}^Adam
    with beta_t = 1 / (local_step + 1)**power. The anchor is a snapshot of
    the parameters taken the first time this function sees them.

    Setting `reset_every` (in epochs) re-snapshots the anchor and clears
    the Adam moments periodically instead of anchoring to the very first
    step forever - this is the "anchor-reset schedule" used for full-scale
    MNIST training. `steps_per_epoch` is required whenever `reset_every`
    is set, since resets are counted in optimizer steps.
    Leaving `reset_every=None` (the default) anchors to the initial
    parameters for the whole run and never resets.
    """
    if reset_every is not None:
        cycle_length = reset_every * steps_per_epoch
        local_step = ((step - 1) % cycle_length) + 1
        is_reset = (step > 1) and (local_step == 1)
    else:
        local_step = step
        is_reset = False

    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            if i not in moments:
                moments[i] = {'m': torch.zeros_like(p.data),
                              'v': torch.zeros_like(p.data),
                              'anchor': p.data.clone()}

            if is_reset:
                moments[i]['anchor'] = p.data.clone()
                moments[i]['m'] = torch.zeros_like(p.data)
                moments[i]['v'] = torch.zeros_like(p.data)

            m_prev, v_prev = moments[i]['m'], moments[i]['v']
            anchor = moments[i]['anchor']

            g = p.grad
            m_new = beta1 * m_prev + (1 - beta1) * g
            v_new = beta2 * v_prev + (1 - beta2) * (g * g)
            moments[i]['m'], moments[i]['v'] = m_new, v_new

            # Adam bias correction uses the global step...
            m_hat = m_new / (1 - beta1 ** step)
            v_hat = v_new / (1 - beta2 ** step)
            update = m_hat / (torch.sqrt(v_hat) + eps)

            if is_discriminator:
                p.data = p.data + lr * update
            else:
                p.data = p.data - lr * update

            # ...but the anchor pull uses the local (post-reset) step.
            beta_t = 1.0 / (local_step + 1) ** power
            p.data = beta_t * anchor + (1.0 - beta_t) * p.data

            p.grad.zero_()


def anchored_optimistic_adam_update(params, moments, step, lr, beta1=0.5, beta2=0.999, eps=1e-8,
                                     power=1, reset_every=None, steps_per_epoch=None,
                                     is_discriminator=False, **kwargs):
    """Optimistic Adam + Halpern anchoring stacked together.

    Combines the look-ahead extrapolation of `optimistic_adam_update` with
    the drift control of `anchored_adam_update`. See both docstrings above
    for what each mechanism does; `power` and `reset_every` behave exactly
    as in `anchored_adam_update`.
    """
    if reset_every is not None:
        cycle_length = reset_every * steps_per_epoch
        local_step = ((step - 1) % cycle_length) + 1
        is_reset = (step > 1) and (local_step == 1)
    else:
        local_step = step
        is_reset = False

    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            if i not in moments:
                moments[i] = {'m': torch.zeros_like(p.data),
                              'v': torch.zeros_like(p.data),
                              'prev_update': torch.zeros_like(p.data),
                              'anchor': p.data.clone()}

            if is_reset:
                moments[i]['anchor'] = p.data.clone()
                moments[i]['m'] = torch.zeros_like(p.data)
                moments[i]['v'] = torch.zeros_like(p.data)
                moments[i]['prev_update'] = torch.zeros_like(p.data)

            m_prev = moments[i]['m']
            v_prev = moments[i]['v']
            u_prev = moments[i]['prev_update']
            anchor = moments[i]['anchor']

            g = p.grad
            m_new = beta1 * m_prev + (1 - beta1) * g
            v_new = beta2 * v_prev + (1 - beta2) * (g * g)
            moments[i]['m'], moments[i]['v'] = m_new, v_new

            m_hat = m_new / (1 - beta1 ** step)
            v_hat = v_new / (1 - beta2 ** step)
            u_current = m_hat / (torch.sqrt(v_hat) + eps)
            optimistic_u = 2 * u_current - u_prev
            moments[i]['prev_update'] = u_current.clone()

            if is_discriminator:
                p.data = p.data + lr * optimistic_u
            else:
                p.data = p.data - lr * optimistic_u

            beta_t = 1.0 / (local_step + 1) ** power
            p.data = beta_t * anchor + (1.0 - beta_t) * p.data

            p.grad.zero_()


# --------------------------------------------------------------------------
# Learning-rate schedules. Each factory returns a callable lr_fn(step) with
# `.initial_lr` and `.name` attached (used for TensorBoard hparam logging).
# `train()` accepts gamma/eta as either a plain float or one of these.
# --------------------------------------------------------------------------

def make_constant_lr(lr_0):
    """Identity schedule, so 'constant' still shows up uniformly in hparams."""
    def lr_fn(t):
        return lr_0
    lr_fn.initial_lr = lr_0
    lr_fn.name = f'constant({lr_0})'
    return lr_fn


def make_inv_sqrt_lr(lr_0, tau):
    """lr_t = lr_0 / sqrt(1 + t / tau). Canonical stochastic primal-dual schedule."""
    def lr_fn(t):
        return lr_0 / (1.0 + t / tau) ** 0.5
    lr_fn.initial_lr = lr_0
    lr_fn.name = f'inv_sqrt(lr0={lr_0}, tau={tau})'
    return lr_fn


def make_inv_linear_lr(lr_0, tau):
    """lr_t = lr_0 / (1 + t / tau). Faster decay than inv-sqrt."""
    def lr_fn(t):
        return lr_0 / (1.0 + t / tau)
    lr_fn.initial_lr = lr_0
    lr_fn.name = f'inv_linear(lr0={lr_0}, tau={tau})'
    return lr_fn


def make_step_lr(lr_0, decay_every_steps, decay_factor=0.5):
    """Multiply lr by decay_factor every `decay_every_steps`. Standard GAN recipe."""
    def lr_fn(t):
        return lr_0 * (decay_factor ** (t // decay_every_steps))
    lr_fn.initial_lr = lr_0
    lr_fn.name = f'step(lr0={lr_0}, every={decay_every_steps}, factor={decay_factor})'
    return lr_fn
