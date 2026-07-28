#include <stdint.h>
#include <nvtx3/nvToolsExt.h>

uint64_t candle_sam3_nvtx_range_start(const char *message) {
  return (uint64_t)nvtxRangeStartA(message);
}

void candle_sam3_nvtx_range_end(uint64_t range_id) {
  nvtxRangeEnd((nvtxRangeId_t)range_id);
}
