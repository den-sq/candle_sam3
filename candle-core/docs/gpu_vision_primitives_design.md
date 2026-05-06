# GPU Vision Primitives And Backend Gaps In Candle

## Goal

Document the Candle-side additions that look broadly useful for GPU-first vision workloads after
profiling a real segmentation/video-tracking model stack.

The focus here is on reusable backend and framework capabilities in Candle itself, not on
SAM3-specific patches in `candle-transformers`.

## Measurement Basis

The current priority order in this note is based on:

- `perf` for CPU-side call attribution
- `nsys` broad and focused GPU traces from the cuDNN-enabled build
- `ncu` only as a qualitative guide from the non-cuDNN build

That `ncu` caveat matters: on this WSL2 setup, `ncu` against the cuDNN build currently fails
during CUDA device enumeration inside cuDNN initialization, so `perf` + `nsys` are the
authoritative measurements for the actual runtime, while `ncu` is only used to understand broad
kernel families and launch patterns.

## Why This Matters

The current GPU profile shape is not "one bad GEMM" or "one missing fusion."

The dominant pattern is:

- many small CUDA kernel launches
- heavy host-to-device helper traffic
- large command-buffer pressure
- norm-heavy and elementwise-heavy vision blocks
- layout/materialization churn around convolution and attention boundaries

That pattern is common well beyond SAM3. Similar issues show up in:

- segmentation decoders
- UNet-like image models
- ConvNeXt-style vision blocks
- ViTs with frequent layout changes
- prompt/feature packing pipelines

The useful response is therefore a small set of general Candle primitives and backend upgrades that
reduce kernel count, reduce materialization, and support common vision layouts directly.

## Current Priority Summary

The highest-value Candle additions now appear to be:

1. Channels-first fused normalization.
2. Grouped/depthwise convolution fast paths.
3. Better small-conv and conv-transpose dispatch, including less reliance on generic `im2col`.
4. Batched tensor-assembly / packing primitives.
5. Layout-aware matmul and attention-input support.
6. Connected-components and component-statistics primitives for mask cleanup.

Items `1` through `5` are the main performance priorities. Item `6` is still useful and still
removes an unnecessary GPU/CPU roundtrip, but it is no longer the top performance lever.

## Proposed Layering

### Low-Level Layer

In `candle-kernels` and the `candle-core` CUDA backend, add:

- reusable kernels and dispatch helpers
- shape/layout-aware routing
- internal ops and fast paths that benefit multiple models

### Framework-Level Layer

In `candle-nn::ops`, add ergonomic wrappers where a primitive is likely to be directly useful to
model authors, especially for:

- channels-first normalization
- mask cleanup
- efficient tensor assembly

The public API should stay general and not mention SAM3.

## Proposed Additions

### 1. Channels-First Fused Normalization

#### What To Add

A CUDA primitive with `LayerNorm2d` semantics on channel-first inputs, without forcing each model
to choose between:

- a custom NCHW elementwise chain, or
- `NCHW -> NHWC -> NCHW` materialization just to use last-dimension `LayerNorm`

Candidate API:

```rust
layer_norm_nchw(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor>
```

Possible follow-ons:

- affine and non-affine variants
- optional fused residual/bias/activation helpers where they map well to backend kernels

#### Why It Is Generally Useful

Channels-first normalization shows up in:

- ConvNeXt-style blocks
- segmentation decoders
- vision necks and fusers
- UNet-like architectures
- image restoration / diffusion backbones

This is a broadly useful primitive because many vision stacks stay NCHW for convolution, and
repeated layout conversion is often worse than the norm math itself.

#### How SAM3 Would Use It

SAM3 still spends a large amount of time in the custom `LayerNorm2d` path inside the mask-memory
encoder/downsampler. A real `layer_norm_nchw` would directly target that hotspot without repeating
the failed "wrap everything in NHWC LayerNorm" approach.

### 2. Grouped And Depthwise Convolution Fast Paths

#### What To Add

Better backend handling for grouped and especially depthwise convolution, so the grouped case does
not degenerate into:

- chunk input by group
- chunk weights by group
- run one conv per group
- concatenate results

This is mostly a backend-routing and kernel-coverage problem rather than a public API problem.

Potential internal capabilities:

- direct grouped-conv kernels for common low-batch shapes
- specialized depthwise conv kernels
- better routing to library kernels when available

#### Why It Is Generally Useful

Grouped and depthwise conv are common in:

- ConvNeXt/MobileNet-style blocks
- segmentation fusers
- lightweight decoders
- edge/mobile inference stacks

Treating grouped conv as repeated single-group convs is bad for launch count, memory traffic, and
command-buffer pressure.

#### How SAM3 Would Use It

SAM3's mask-memory fuser uses depthwise `7x7` convolution in a hot path. Current Candle grouped
conv behavior makes that more expensive than it should be even when other pieces are improved.

### 3. Better Small-Conv And Conv-Transpose Dispatch

#### What To Add

Improve CUDA dispatch and kernel coverage for conv families that are common in segmentation and
vision inference:

- pointwise `1x1` conv
- small `3x3` conv
- small stride-2 downsampling conv
- common low-batch transposed conv shapes
- common low-batch segmentation conv workloads

This does not necessarily require a new public API. It may mostly be:

- backend routing
- shape-specialized kernels
- better heuristics
- less reliance on generic `im2col`

#### Why It Is Generally Useful

Generic `im2col` is convenient, but it is often not the best steady-state path for modern vision
inference with small or medium spatial shapes and low batch sizes.

These shapes show up in:

- segmentation heads
- decoder upscalers
- feature pyramid fusers
- prompt-conditioned image models

#### How SAM3 Would Use It

This would target the conv-heavy mask-memory encoder and the decoder-side upscaling path, both of
which still show generic lowering and helper-kernel overhead in profiling.

### 4. Batched Tensor-Assembly Primitives

#### What To Add

Low-level primitives for assembling many source tensors into one destination with one launch or a
small bounded set of launches, instead of one copy kernel per input slice.

Candidate operations:

```rust
cat_many(tensors: &[&Tensor], dim: usize) -> Result<Tensor>
stack_many(tensors: &[&Tensor], dim: usize) -> Result<Tensor>
copy_many_2d(specs: &[Copy2DSpec], dst: &Tensor) -> Result<()>
pack_by_index(src: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor>
```

These could be:

- public ops, or
- internal assembly primitives used by `cat`, `stack`, `index_select`, and common packing paths

#### Why It Is Generally Useful

Current GPU inference workloads often spend surprising time in:

- concat / stack
- tensor packing
- `contiguous()`
- repeated copy kernels launched in sequence

This affects:

- transformer prompt assembly
- decoder token packing
- multi-branch feature fusion
- autoregressive cache maintenance
- vision feature-pyramid assembly

#### How SAM3 Would Use It

SAM3 has already been cleaned up substantially at the model layer, but `copy2d` and `ucopy` style
bursts still appear in decoder, prompt-history, and attention-adjacent packing paths.

### 5. Layout-Aware Matmul And Attention Inputs

#### What To Add

Better support for consuming common transposed or head-split layouts directly, instead of forcing
model code to materialize `transpose(...).contiguous()` before matmul.

Possible directions:

- support for selected strided batched-matmul layouts
- internal helpers for attention-style head split / recombination
- layout-aware dispatch that materializes once at a strategic boundary instead of at every call

#### Why It Is Generally Useful

Transformers and ViTs routinely pay extra cost for:

- head splitting
- Q/K/V transposes
- recombination after attention
- small batched GEMV/GEMM calls on awkward layouts

This affects:

- language models
- vision transformers
- multimodal encoders
- decoder cross-attention blocks

#### How SAM3 Would Use It

This would reduce the remaining materialization cost in SAM decoder attention and some vision
attention paths, where local model-only rewrites have mostly run out of headroom.

### 6. Connected Components And Component Stats

#### What To Add

Low-level primitives for:

```rust
connected_components_2d(mask: &Tensor, connectivity: Connectivity) -> Result<Tensor>
component_sizes_2d(labels: &Tensor) -> Result<Tensor>
component_border_flags_2d(labels: &Tensor) -> Result<Tensor>
rewrite_components_2d(
    src: &Tensor,
    labels: &Tensor,
    selected: &Tensor,
    replacement: &Tensor,
) -> Result<Tensor>
```

Higher-level wrappers in `candle-nn::ops` for:

- `fill_small_holes_2d`
- `remove_small_connected_components_2d`
- `cleanup_mask_logits_small_components_2d`

#### Why It Is Generally Useful

This is a standard topology-aware vision primitive useful for:

- segmentation cleanup
- OCR/document cleanup
- medical and biological mask filtering
- instance-mask preprocessing
- topology-aware postprocessing

It belongs in Candle because it cannot be expressed efficiently as a simple chain of existing
pointwise ops.

#### How SAM3 Would Use It

This would remove the remaining exact GPU -> CPU -> GPU cleanup roundtrip in SAM3 video
postprocessing while preserving current semantics.

## Supporting Backend/Runtime Work

These are not public primitives, but they are still worth calling out.

### CUDA Event Pooling And Lighter Bookkeeping

Current profiling still shows meaningful time in:

- `cuEventRecord`
- event create/destroy
- command-buffer-full overhead

Reducing per-op event churn and reusing events more aggressively would benefit many workloads once
the main model-side small-kernel count has been pushed down.

### Better Library Dispatch Visibility

When optional libraries such as cuDNN are available, it is useful for Candle to expose or log which
backend path was selected for conv-heavy ops. That would make performance diagnosis much easier for
users without requiring `ncu`.

## Why Existing Candle Ops Are Not Enough

Current Candle building blocks are strong for:

- pointwise math
- reductions
- generic convolution
- indexing
- last-dimension `LayerNorm`

The main remaining gaps are specifically about:

- channels-first normalization
- grouped/depthwise convolution
- avoiding generic `im2col` for small repeated convs
- high-fan-in tensor assembly
- reducing layout materialization around attention
- topology-aware mask cleanup

These are framework/backend issues rather than SAM3-specific modeling logic.

## Testing Requirements

The Candle implementation should include:

1. Correctness tests against simple CPU references.
2. Shape-coverage tests for common low-batch vision cases.
3. Randomized parity tests for CUDA and CPU where a CPU fallback exists.
4. Performance smoke tests on representative shapes so future regressions are easier to catch.
5. Dispatch-path tests where a feature is mainly a routing improvement.

For connected components specifically:

- all-background
- all-foreground
- isolated holes/sprinkles
- border-touching cases
- batched multi-plane inputs

For grouped/depthwise conv specifically:

- grouped conv with non-trivial group counts
- pure depthwise conv
- low-batch shape coverage
- parity against the existing generic implementation

## Non-Goals

This proposal does not require:

- baking SAM3-specific policy into Candle public APIs
- changing model semantics in downstream crates
- replacing every small kernel with one giant fused kernel

It does aim to close the most important low-level gaps that repeatedly show up across GPU-first
vision inference workloads.

## Recommendation Summary

The highest-value Candle additions are now:

1. channels-first fused normalization
2. grouped/depthwise conv fast paths
3. better small-conv / conv-transpose dispatch
4. batched tensor assembly primitives
5. layout-aware matmul/attention support
6. connected-components and component-statistics support

That set would improve Candle broadly for segmentation, OCR, medical imaging, ViTs, and other
GPU-first vision workloads, while also addressing the main remaining performance bottlenecks seen
in SAM3 without making the framework SAM3-specific.
