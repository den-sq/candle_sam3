/*
 * The attention kernel is derived from PyTorch 2.7's memory-efficient
 * attention implementation:
 * aten/src/ATen/native/transformers/cuda/mem_eff_attention.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the BSD-style license carried by the vendored headers.
 */

#include <cuda_runtime.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

using CandleF32Sm75Attention =
    PyTorchMemEffAttention::AttentionKernel<
        float,
        cutlass::arch::Sm75,
        true,
        64,
        64,
        64,
        true,
        true>;

__global__ void __launch_bounds__(
    CandleF32Sm75Attention::kNumThreads,
    CandleF32Sm75Attention::kMinBlocksPerSm)
candle_f32_sm75_attention(CandleF32Sm75Attention::Params params) {
  if (!params.advance_to_block()) {
    return;
  }
  CandleF32Sm75Attention::attention_kernel(params);
}

extern "C" int launch_candle_f32_sm75_attention(
    const float *query,
    const float *key,
    const float *value,
    float *output,
    int batch,
    int heads,
    int query_sequence,
    int key_sequence,
    float scale,
    cudaStream_t stream) {
  int device;
  int major;
  int minor;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  error = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  error = cudaDeviceGetAttribute(
      &minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  if (major != 7 || minor != 5) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  CandleF32Sm75Attention::Params params;
  params.query_ptr = query;
  params.key_ptr = key;
  params.value_ptr = value;
  params.output_ptr = output;
  params.output_accum_ptr = nullptr;
  params.logsumexp_ptr = nullptr;
  params.num_batches = batch;
  params.num_heads = heads;
  params.num_queries = query_sequence;
  params.num_keys = key_sequence;
  params.num_keys_absolute = key_sequence;
  params.head_dim = 64;
  params.head_dim_value = 64;
  params.scale = scale;

  // Candle tensors use contiguous BHSD layout. The upstream kernel accepts
  // arbitrary sequence/head/batch strides.
  params.q_strideM = 64;
  params.k_strideM = 64;
  params.v_strideM = 64;
  params.o_strideM = 64;
  params.o_strideH = query_sequence * 64;
  params.o_strideB =
      static_cast<int64_t>(heads) * query_sequence * 64;
  params.q_strideH = query_sequence * 64;
  params.k_strideH = key_sequence * 64;
  params.v_strideH = key_sequence * 64;
  params.q_strideB =
      static_cast<int64_t>(heads) * query_sequence * 64;
  params.k_strideB =
      static_cast<int64_t>(heads) * key_sequence * 64;
  params.v_strideB =
      static_cast<int64_t>(heads) * key_sequence * 64;

  constexpr size_t shared_memory =
      sizeof(CandleF32Sm75Attention::SharedStorage);
  if (shared_memory > 0xc000) {
    error = cudaFuncSetAttribute(
        candle_f32_sm75_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory);
    if (error != cudaSuccess) {
      return static_cast<int>(error);
    }
  }
  const dim3 grid(
      (query_sequence + CandleF32Sm75Attention::kQueriesPerBlock - 1) /
          CandleF32Sm75Attention::kQueriesPerBlock,
      heads,
      batch);
  candle_f32_sm75_attention<<<
      grid,
      dim3(32, CandleF32Sm75Attention::kNumThreads / 32, 1),
      shared_memory,
      stream>>>(params);
  return static_cast<int>(cudaGetLastError());
}
