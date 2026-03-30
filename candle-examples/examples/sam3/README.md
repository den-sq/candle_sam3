# SAM 3 Example

This example now runs the implemented SAM3 image pipeline end to end:

1. load `sam3.pt`
2. resize the image to `1008x1008` by default
3. normalize with mean/std `0.5`
4. call `set_image`
5. call `set_text_prompt` through the typed image state
6. tokenize and encode the text prompt
7. run the ViTDet trunk, FPN neck, fusion encoder, DETR decoder, and segmentation head
8. render the best predicted box and mask to disk

The rendered outputs are:

- `overlay.png`: resized square input with the predicted mask overlay and predicted box
- `mask.png`: grayscale mask confidence image for the selected query, bilinearly upsampled to the render image size
- `summary.json`: prompt, score, normalized box, and output paths

If `--box` prompts are provided, they are:

- stored in the typed state
- encoded by the geometry encoder smoke path
- drawn in blue on `overlay.png`

The predicted box is drawn in green.

## Relevant modules

- `sam3/text.rs`: CLIP-like text encoder + resize projection
- `sam3/vitdet.rs`: ViTDet visual trunk
- `sam3/neck.rs`: FPN projection and position encodings
- `sam3/encoder.rs`: visual-language fusion encoder
- `sam3/decoder.rs`: DETR-style decoder with presence-token scoring
- `sam3/segmentation.rs`: MaskFormer-style segmentation head
- `sam3/image.rs`: typed image API used by the example

## Checkpoint and tokenizer paths

The example accepts either:

- a direct path to `sam3.pt`
- a repo directory containing `sam3.pt`

And either:

- a direct path to `tokenizer.json`
- a repo directory containing `tokenizer.json`

## Default image pipeline

Unless `--smoke-image-size` is set, the example uses the model default image size:

- resize to `1008x1008`
- normalize RGB with mean `[0.5, 0.5, 0.5]`
- normalize RGB with std `[0.5, 0.5, 0.5]`

`--smoke-image-size` is still available for faster CPU runs while debugging, but the
default path is the real `1008x1008` pipeline.

## CPU Example

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/output
```

## CPU Example With Intel oneAPI MKL

Source the oneAPI MKL environment first so `intel-mkl-src` can locate the
system MKL installation instead of falling back to the downloaded `ocipkg`
package cache.

```bash
source /opt/intel/oneapi/mkl/2025.3/env/vars.sh

cargo run -p candle-examples --example sam3 --features mkl -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/output
```

You can also source `/opt/intel/oneapi/setvars.sh` instead if you want the full
oneAPI environment rather than MKL alone.

## CPU Example With Box Prompt Rendering

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --box 0.5,0.5,0.45,0.45 \
  --output-dir candle-examples/examples/sam3/output
```

## CUDA Example

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
cargo run -p candle-examples --example sam3 --features cuda -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/output
```

Update environment variable paths to the appropriate ones for your CUDA version.  If your environment blocks automatic compute-capability detection during CUDA builds, set `CUDA_COMPUTE_CAP` as well.

## CUDA Example With Intel oneAPI MKL

```bash
source /opt/intel/oneapi/mkl/2025.3/env/vars.sh

PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
cargo run -p candle-examples --example sam3 --features cuda,mkl -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/output
```

If you source `/opt/intel/oneapi/mkl/2025.3/env/vars.sh` first, `LD_LIBRARY_PATH`
will already contain the MKL runtime directory. The inline CUDA override above
keeps that existing value by appending `:$LD_LIBRARY_PATH`.

## Parity Harness

Step 11 is implemented as a stage-by-stage parity harness. It compares the Rust
pipeline against a reference bundle exported from upstream PyTorch for one fixed
image/prompt pair.

The parity bundle contains:

- `inputs.image`: preprocessed `1x3xHxW` image tensor
- `inputs.input_ids`: token IDs used by upstream SAM3
- `inputs.attention_mask`: token attention mask
- `text.input_embeddings`
- `text.memory`
- `vision.backbone_fpn.0..N`
- `fusion.memory`
- `decoder.pred_logits`
- `decoder.pred_boxes_xyxy`
- `segmentation.mask_logits`

Optional debug tensors such as `fusion.pos_embed`, `decoder.presence_logits`, and
`segmentation.semantic_logits` are exported too when present.

### Export Reference Bundle From Upstream PyTorch

The exporter script assumes you have a Python environment with:

- `torch`
- `torchvision`
- `safetensors`
- `Pillow`

And a local checkout of the official `facebookresearch/sam3` repo.

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_den/sam3 \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/reference
```

This writes:

- `candle-examples/examples/sam3/reference/reference.safetensors`
- `candle-examples/examples/sam3/reference/reference.json`

### Run Rust Parity Check

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference \
  --output-dir candle-examples/examples/sam3/output
```

The parity report is written to:

- `candle-examples/examples/sam3/output/parity_report.json`

By default the harness fails if any stage exceeds `1e-4` absolute error. You can
override that with `--parity-atol 1e-3`.

## Notes

- The example prints intermediate stage shapes so it still works as a debug harness.
- Geometry mask prompts are still not available because the current checkpoint does not
  include a geometry mask-encoder branch.
- Full video / tracker support is still outside this example.
