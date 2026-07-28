#include <cuda_runtime.h>

extern "C" int launch_candle_f32_sm75_attention(
    const float *,
    const float *,
    const float *,
    float *,
    int,
    int,
    int,
    int,
    float,
    cudaStream_t) {
  return static_cast<int>(cudaErrorNotSupported);
}
