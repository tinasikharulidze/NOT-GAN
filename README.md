# Neural Optimal Transport GANs

This repo implements a primal-dual formulation of Optimal Transport (OT) as
a generative modeling framework, and applies it progressively to three
settings of increasing difficulty: a 2D toy distribution, MNIST digits
(unconditional and class-conditional), and CartoonSet faces conditioned on
free-form text. It's the codebase behind an MSc thesis (Tina Sikharulidze &
Anabel Pichardo, supervised by Gergely Neu).

## The idea, in short

Optimal Transport asks: what's the cheapest way to reshape one distribution
of mass into another? Framed as a generative model, the "source"
distribution is noise and the "target" is real data (digits, faces, ...),
and the model learns a map that pushes noise onto data as efficiently as
possible.

The specific formulation used here comes from a Lagrangian relaxation of
the OT problem, which turns out to look a lot like a GAN: a **generator**
(primal variable) tries to move source points to target points as cheaply
as possible, while a **critic** (dual variable, an unconstrained neural
network - no Lipschitz constraint, no gradient penalty, no weight clipping)
prices how far the current transport plan is from matching the real data
distribution. Concretely:

- Critic loss (maximize): `f(Y_real) − f(Y_fake)`
- Generator loss (minimize): `‖X − Y_fake‖² − f(Y_fake)`

The first term is the actual transport cost - a generated sample is
penalized for straying far from the noise it came from. This is the main
thing that distinguishes this approach from a standard GAN (e.g. WGAN):
there's no explicit notion of "cost" in a normal GAN, only "does this look
real". That cost term turns out to have a nice side effect: if you feed a
*real image* in as the "noise" at inference time, the model naturally edits
it towards the prompt rather than regenerating from scratch - image-to-image
translation falls out for free, with no architectural change (see the
CartoonSet section below).

Because the critic is unconstrained, this min-max game is numerically
unstable in the way GANs generally are, just more so - and a large part of
this project (and this codebase) is about the optimizer machinery that
makes it converge anyway: Adam variants with **optimistic** look-ahead
(damps oscillation) and **Halpern anchoring** (periodically pulls parameters
back toward a reference point, which provably bounds the gradient norm in
this kind of saddle-point problem).

## The experiments, and what they found

Each experiment builds on lessons from the previous one. Rough numbers
below (see the thesis for full tables/figures):

**1. 2D toy data** (`2D_Experiments.ipynb`) - a 2D Gaussian source and a
3-component Gaussian-mixture target, small enough to plot the actual
transport map. Used to sanity-check the algorithm and sweep learning rate,
optimizer, noise level, and batch size before touching images. Established
**Anchored Optimistic Adam** as the default optimizer for everything after
this.

**2. MNIST, unconditional** (`MNIST_Training.ipynb`) - moving to image data
immediately exposed a failure mode that doesn't show up in 2D: sampling
noise directly in the full pixel space (e.g. 32×32 = 1024 dimensions) causes
a concentration-of-measure effect where every noise vector is nearly
equidistant from every other, so the generator collapses to emitting the
single pixel-mean image. Fixed by sampling **low-frequency noise**
(an 8×8 random draw, bilinearly upsampled to full resolution) instead of
full-resolution noise. Combined with anchoring to bound the critic's
gradients, this gets a stable, non-collapsed unconditional model to
**FID ≈ 20.3**. 

**3. MNIST, class-conditional** (also `MNIST_Training.ipynb`) - adding a
class label (via a projection critic + label embedding in the generator)
turns one hard 10-mode problem into something closer to ten easy one-mode
problems. 

**4. CartoonSet, text-conditioned** (`CartoonSet_Preprocessing.ipynb` +
`CartoonSet_Training.ipynb`) - the final and most ambitious setting: 64×64
color cartoon faces, conditioned on natural-language text prompts via CLIP
embeddings, rather than discrete class labels. This required generating
synthetic captions for every image (attribute → randomized natural-language
description), fine-tuning CLIP so it actually separates cartoon attributes
in embedding space (off-the-shelf CLIP was trained on real photos, not
cartoons), and a generator/critic pair conditioned on the resulting
embeddings via Adaptive Group Normalization (AdaGN). Evaluation here is qualitative:
attribute fidelity, generalization to unseen prompts, and - the interesting
emergent property mentioned above - image-to-image translation and smooth
SLERP interpolation between prompts, both free consequences of the explicit
transport cost.

## `src/` - shared library code

All four notebooks import from here. Nothing in `src/` is
notebook-specific; each file is organized by experiment (2D / MNIST /
CartoonSet section) so you can see at a glance what's shared and what's
particular to one setting.

- **`optimizers.py`** - the update rules: `SGD_update`, `adam_update`,
  `optimistic_adam_update`, `anchored_adam_update`,
  `anchored_optimistic_adam_update`, plus a few learning-rate schedule
  helpers. This is the stabilization machinery described above, and it's
  identical across all three experiments.
- **`models.py`** - every generator/critic architecture, grouped by
  experiment: small MLPs for the 2D case, a U-Net generator + conv critic
  for MNIST (plus class-conditional variants with a projection critic), and
  three CLIP-conditioned U-Net generator variants for CartoonSet (they
  differ in *where* the CLIP embedding gets injected - bottleneck-only,
  bottleneck + attention, or fully injected through every encoder/decoder
  block). The fully-injected variant is the one behind the CartoonSet
  results reported in the thesis.
- **`data.py`** - sampling and data-loading: synthetic 2D distributions,
  MNIST loaders, and the CartoonSet datasets/loaders (one for CLIP
  fine-tuning, a different one for GAN training - they return different
  things, see the file itself).
- **`metrics.py`** - training diagnostics (gradient-norm variants used to
  monitor critic stability) and evaluation metrics (Wasserstein-2 distance,
  in an exact and a sliced/approximate flavor depending on image size;
  cluster-coverage; CLIP-embedding attribute-separability audits for
  CartoonSet). FID / precision-recall / density-coverage - the metrics used
  in the thesis's quantitative MNIST tables.
- **`plotting.py`** - every matplotlib figure builder: 2D scatter/transport
  plots, generated-image grids, conditional class grids, CartoonSet dataset
  exploration grids, and CLIP-embedding UMAP cluster plots.
- **`training.py`** - the actual training loops, one function per
  experiment (`train_2d`, `train_mnist`, `train_mnist_cond`,
  `train_mnist_cond_wgan_gp`, `train_clip_cond`), since they differ
  structurally (conditioning, gradient penalties, CFG dropout) even though
  they share the optimizers above. Also holds the CLIP fine-tuning stage
  (`finetune_clip` and helpers) and the CartoonSet experiment launcher
  (`run_cartoon_experiment`) that ties architecture choice, data loading,
  and the training loop together for a given run.
- **`captions.py`** - `generate_clip_caption`, which turns a CartoonSet
  attribute row into a randomized natural-language caption (random synonym
  choice, random attribute ordering, random opener phrase) - this
  randomization is what lets the fine-tuned CLIP encoder generalize to the
  many different ways a real user might phrase the same request.

## `notebooks/` (one per experiment)

Each notebook holds only the setup and orchestration specific to that
experiment; everything reusable comes from `src/`.

- **`2D_Experiments.ipynb`** - self-contained. Defines the source/target
  distributions, then runs four hyperparameter sweeps (learning rate,
  optimizer, noise level, batch size) each as its own section, and ends
  with a results table.

- **`MNIST_Experiment.ipynb`** - organized into
  sections mirroring the thesis's experimental progression: **Model A**
  (early anchoring ablations), **Model B** (the final unconditional model,
  culminating in the FID-20.3 run), **Model C** (class-conditional), and
  **Model CW** (the WGAN-GP baseline), followed by a later addendum of
  optimizer/reset-schedule ablations. Most cells in the earlier sections
  are commented out - they're a kept record of what was tried, not meant to
  be re-run - with the specific cells that produced the reported results
  left active. 

- **`CartoonSet_Preprocessing.ipynb`** - run this before
  `CartoonSet_Experiment.ipynb`. Loads the CartoonSet attribute CSVs,
  generates captions, fine-tunes CLIP, and precomputes an image embedding
  for every image. Produces three files the training notebook needs:
  `cartoon_with_embeddings.csv`, `cartoon_clip_embeddings.npy`, and
  `clip_cartoon_finetuned.pt`.

- **`CartoonSet_Experiment.ipynb`** - loads those three files, sanity-checks
  each architecture variant, and launches the actual training runs (E19,
  E20, E21 - all using the fully-injected architecture). **Requires a CUDA
  GPU**: training wraps the models in `nn.DataParallel` and doesn't fall
  back to CPU.

## Note

- **CartoonSet training needs a GPU; the other three notebooks don't.**
