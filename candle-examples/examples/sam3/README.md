# SAM 3 Example

This example now runs the implemented SAM3 image pipeline end to end:

1. load `sam3.pt`
2. preprocess the image into the model's `1008x1008` input by default
3. normalize with mean/std `0.5`
4. call `set_image`
5. call `set_text_prompt` through the typed image state
6. tokenize and encode the text prompt
7. run the ViTDet trunk, FPN neck, fusion encoder, DETR decoder, and segmentation head
8. render the best predicted box and mask to disk
9. repeat the same sample with both `exact` and `crop_fill` preprocessing

The rendered outputs are:

- `exact/` and `crop_fill/`: one subdirectory per preprocessing mode for each sample
- `overlay.png`: rendered on the original image aspect ratio; by default this is the prediction overlay, while `--image-predictor-example` box jobs use notebook-style prompt visualization here
- `prediction_overlay.png`: prediction overlay with the selected mask and box on the original image
- `mask.png`: grayscale `sigmoid(mask_logits)` image used by the default overlay path
- `mask_sigmoid.png`: explicit grayscale `sigmoid(mask_logits)` dump
- `mask_one_minus_sigmoid.png`: explicit grayscale `1 - sigmoid(mask_logits)` dump
- `mask_sigmoid_threshold_0_5.png`: binary thresholded mask from `sigmoid(mask_logits)`
- `mask_one_minus_sigmoid_threshold_0_5.png`: binary thresholded mask from `1 - sigmoid(mask_logits)`
- `prediction_overlay_sigmoid_threshold_0_5.png`: thresholded overlay using `sigmoid(mask_logits)`
- `prediction_overlay_one_minus_sigmoid_threshold_0_5.png`: thresholded overlay using `1 - sigmoid(mask_logits)`
- `summary.json`: prompt, score, normalized box, and output paths

If point or box prompts are provided, they are:

- stored in the typed state
- encoded by the geometry encoder smoke path
- drawn on `overlay.png`

Prompt colors:

- positive points / boxes: blue
- negative points / boxes: red
- predicted box and mask: green

For `--image-predictor-example`, the box-only notebook replay uses upstream-style prompt colors instead:

- positive boxes: green
- negative boxes: red

## Notebook Coverage

The downloaded upstream notebooks split into three groups:

- supported now:
  - `sam3_image_predictor_example.ipynb`
  - `sam3_image_batched_inference.ipynb`
- partially mapped:
  - `sam3_image_interactive.ipynb`
- not yet supported:
  - `sam3_for_sam1_task_example.ipynb`
  - `sam3_for_sam2_video_task_example.ipynb`
  - `sam3_video_predictor_example.ipynb`
  - `sam3_agent.ipynb`

What is implemented in this Candle example:

- text-only image grounding
- geometry-only image grounding with positive and negative points / boxes
- mixed text + geometry grounding
- sequential batch-manifest execution that mirrors the batched notebook workflow at the CLI level
- a canned `--image-predictor-example` mode that replays the inputs from `sam3_image_predictor_example.ipynb`

What is not implemented here yet:

- Jupyter widget interaction
- SAM1-style interactive mask refinement
- video session state and tracking
- agent / LLM integration

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

Unless `--smoke-image-size` is set, each sample is run twice against the model default image size:

- preprocess into a `1008x1008` model input tensor with `exact`
- preprocess into a `1008x1008` model input tensor with `crop_fill`
- normalize RGB with mean `[0.5, 0.5, 0.5]`
- normalize RGB with std `[0.5, 0.5, 0.5]`

`--smoke-image-size` is still available for faster CPU runs while debugging, but the
default path now compares both preprocessing modes side by side.

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

## CPU Geometry-Only Example

This is the closest CLI equivalent to the visual-prompt sections of
`sam3_image_predictor_example.ipynb`.

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --box 0.48,0.53,0.72,0.78 \
  --box-label 1 \
  --point 0.48,0.50 \
  --point-label 1 \
  --output-dir candle-examples/examples/sam3/output
```

## CPU Mixed Text + Geometry Example

This matches the notebook workflow where a text prompt is refined by positive or
negative geometric prompts.

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "cat" \
  --box 0.48,0.53,0.72,0.78 \
  --box-label 1 \
  --box 0.16,0.20,0.20,0.20 \
  --box-label 0 \
  --point 0.48,0.50 \
  --point-label 1 \
  --point 0.12,0.15 \
  --point-label 0 \
  --output-dir candle-examples/examples/sam3/output
```

## Canned Image Predictor Notebook Replay

This runs the built-in replay of the upstream
`examples/sam3_image_predictor_example.ipynb` scenarios on the original
`test_image.jpg`:

- text prompt `shoe`
- one positive box prompt
- one positive + one negative box prompt

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image-predictor-example \
  --output-dir candle-examples/examples/sam3/output/image_predictor_example
```

Each notebook scenario writes into its own subdirectory under the requested
output directory, and each scenario then writes `exact/` and `crop_fill/`
subdirectories underneath that. For the two box scenarios:

- `overlay.png` matches the notebook prompt-box visualization
- `prediction_overlay.png` contains the model prediction overlay on the original image

## Sequential Batch-Manifest Example

This is the CLI equivalent of `sam3_image_batched_inference.ipynb`. The current
Candle example executes jobs sequentially rather than packing them into one
training-style upstream batch, but it supports the same mix of text-only,
geometry-only, and mixed jobs.

Use the included manifest:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --batch-manifest candle-examples/examples/sam3/batch_manifest.example.json \
  --output-dir candle-examples/examples/sam3/output/batch
```

Each manifest job writes into its own subdirectory under the requested output
directory.

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

## CUDA Mixed Prompt Example

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
cargo run -p candle-examples --example sam3 --features cuda -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "cat" \
  --box 0.48,0.53,0.72,0.78 \
  --box-label 1 \
  --box 0.16,0.20,0.20,0.20 \
  --box-label 0 \
  --point 0.48,0.50 \
  --point-label 1 \
  --point 0.12,0.15 \
  --point-label 0 \
  --output-dir candle-examples/examples/sam3/output
```

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
- `inputs.boxes_cxcywh`: optional model-space box prompts in normalized `cx,cy,w,h`
- `inputs.box_labels`: optional boolean box labels aligned with `inputs.boxes_cxcywh`
- `text.input_embeddings`
- `text.memory`
- `vision.backbone_fpn.0..N`
- `geometry.features`
- `geometry.padding_mask`
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
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --output-dir candle-examples/examples/sam3/reference
```

This writes:

- `candle-examples/examples/sam3/reference/reference.safetensors`
- `candle-examples/examples/sam3/reference/reference.json`

That is the true upstream parity mode. It uses the same exact square resize as
the Python SAM3 image processor.

### Export Diagnostic Crop-Fill Parity Bundle

If you want to compare Candle's non-upstream `crop_fill` example mode against a
matching Python-side bundle, export it explicitly:

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --prompt "a cat" \
  --box 0.5,0.5,0.45,0.45 \
  --box-label 1 \
  --preprocess-mode crop_fill \
  --output-dir candle-examples/examples/sam3/reference_crop_fill
```

In this mode, the exporter:

- preprocesses the image with `crop_fill`
- remaps box prompts into crop-space before encoding
- writes those remapped model-space boxes into the bundle

This is a local diagnostic mode for the Candle example, not the upstream SAM3
preprocessing contract.

### Run Rust Parity Check

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference \
  --output-dir candle-examples/examples/sam3/output
```

The same command works with a diagnostic crop-fill bundle too:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference_crop_fill \
  --output-dir candle-examples/examples/sam3/output_crop_fill
```

The parity report is written to:

- `candle-examples/examples/sam3/output/parity_report.json`

By default the harness fails if any stage exceeds `1e-4` absolute error. You can
override that with `--parity-atol 1e-3`.

## Notes

- The example prints intermediate stage shapes so it still works as a debug harness.
- Geometry mask prompts are still not available because the current checkpoint does not
  include a geometry mask-encoder branch.
- The batch manifest is a sequential inference runner, not a true one-pass multi-image batch.
- Full video / tracker support is still outside this example.
