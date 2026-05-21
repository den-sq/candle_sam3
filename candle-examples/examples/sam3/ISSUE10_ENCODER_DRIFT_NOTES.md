# Issue 10 Encoder Drift Notes

This note records the tradeoff behind the issue-10 CPU vision-trunk fix. The
bug showed up as a SAM3 parity failure around vision block 8, but the shorter
block-0 debug bundle showed the mismatch was already seeded before the first
trunk block:

- `vision.pre_block.patch_embed`: about `1.431e-6`
- `vision.pre_block.ln_pre`: about `9.537e-6`
- `vision.block.0`: about `1.2398e-5`
- `vision.block.8`: about `4.5776e-5` after the fix

Before the fix, the block-8 branch output drift was above the issue threshold.
After aligning the CPU patch embedding and layer-norm numerics more closely
with the upstream PyTorch reference, the block-8 output stayed below `1e-4`.

## Previous Candle Approach

The previous implementation used Candle's generic CPU kernels:

- tiled im2col/GEMM for the SAM3 patch embedding convolution
- `f32` row reductions for CPU layer norm
- Candle's normal CPU reduction and accumulation orders

Advantages:

- It was simpler and more generic.
- It avoided SAM3-shaped special cases in lower-level kernels.
- The tiled/GEMM convolution path is likely faster for many convolution shapes.
- The results were mathematically valid floating-point results for the same
  model operations.

Disadvantages:

- It was not numerically close enough to the PyTorch CPU reference for this
  SAM3 trunk path.
- Tiny legal floating-point differences were amplified by the ViT residual
  blocks and attention layers.
- The first visible parity failure appeared later in the trunk, which made the
  source look like a block-7 or block-8 bug until earlier bundle captures were
  added.

## Upstream-Parity Approach

The issue-10 fix keeps a narrow CPU path for ViT-style non-overlapping
3-channel patch embeddings and changes CPU `f32` layer-norm row statistics to
use `f64` accumulation before casting the output back to `f32`.

Advantages:

- It matches the upstream PyTorch CPU reference much more closely for SAM3.
- It makes the block-0 seed and downstream compounding visible and explainable.
- It keeps the block-8 vision-trunk drift under the parity threshold.
- It gives a focused regression for the order-sensitive patchify convolution.

Disadvantages:

- It adds a special-case CPU convolution path for a parity-sensitive shape.
- It changes generic CPU `f32` layer-norm numerics, which may have performance
  cost compared with pure `f32` row reductions.
- It follows backend-level PyTorch behavior rather than a model-level semantic
  requirement.
- It should be treated as parity-motivated unless broader benchmarking shows it
  is preferable as a general kernel choice.

## Interpretation

This does not look like a deliberate SAM3 modeling detail in upstream. It looks
like ordinary PyTorch CPU backend behavior: kernel selection, reduction order,
and accumulator precision. Candle's previous implementation was not wrong in a
mathematical sense; it made different legal floating-point choices.

SAM3's vision trunk is sensitive enough that those small choices compound. For
this port, where reproducible parity with upstream is an explicit goal, the
upstream-aligned behavior is justified. The important constraint is to keep the
special convolution path narrow and document it as a parity choice rather than
as a universally more correct convolution implementation.
