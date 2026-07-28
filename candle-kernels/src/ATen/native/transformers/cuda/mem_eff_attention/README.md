# PyTorch memory-efficient attention headers

These headers are derived from
`aten/src/ATen/native/transformers/cuda/mem_eff_attention` in PyTorch
v2.7.0, commit `134179474539648ba7dee1317959529fbd0e7f89`.

The forward-only F32 path was adapted to remove PyTorch runtime dependencies,
to accept Candle's contiguous BHSD output strides, and to build against the
same CUTLASS revision used by that PyTorch release
(`afa1772203677c5118fcd82537a9c8fefbcc7008`).

The source is redistributed under the license in `LICENSE`.
