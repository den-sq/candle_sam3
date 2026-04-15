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

The rendered outputs are:

- `overlay.png`: rendered on the original image aspect ratio; by default this is the prediction overlay, while `--image-predictor-example` box jobs use notebook-style prompt visualization here
- `prediction_overlay.png`: prediction overlay with the selected mask and box on the original image
- `prediction_overlay_all_kept.png`: notebook-style overlay of all kept predictions above the confidence threshold
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
- Phase 12 MVP for deterministic `--interactive` image refinement replays
- Phase 12 scaffolding for `--video` session runs

What is not implemented here yet:

- Jupyter widget interaction
- live GUI-driven interactive refinement UX
- true multi-frame video tracking and object propagation
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

Unless `--smoke-image-size` is set, each sample is run once against the model default image size:

- preprocess into a `1008x1008` model input tensor with `exact`
- normalize RGB with mean `[0.5, 0.5, 0.5]`
- normalize RGB with std `[0.5, 0.5, 0.5]`

`--smoke-image-size` is still available for faster CPU runs while debugging, but the
default path now stays on the parity-validated exact preprocessing path.

The example commands below use the fixed notebook assets from the downloaded
upstream reference checkout under
`/home/dnorthover/extcode/sam3_baseline/assets/images/`. Adjust those paths if
your checkout lives elsewhere.

## CPU Example

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
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
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/output
```

You can also source `/opt/intel/oneapi/setvars.sh` instead if you want the full
oneAPI environment rather than MKL alone.

## CPU Example With Box Prompt Rendering

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
  --box 0.41796875,0.6527777778,0.0859375,0.5 \
  --output-dir candle-examples/examples/sam3/output
```

## CPU Geometry-Only Example

This is the closest CLI equivalent to the point-based image-prediction sections
of `sam3_for_sam1_task_example.ipynb`.

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/truck.jpg \
  --point 0.2888888889,0.3125 \
  --point-label 1 \
  --output-dir candle-examples/examples/sam3/output
```

## CPU Mixed Text + Geometry Example

This uses the same `truck.jpg` point and box locations shown in
`sam3_for_sam1_task_example.ipynb`, with an added text prompt so the CLI still
exercises the mixed text-plus-geometry path.

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/truck.jpg \
  --prompt "truck" \
  --box 0.3125,0.6145833333,0.1527777778,0.2291666667 \
  --box-label 1 \
  --point 0.2888888889,0.3125 \
  --point-label 1 \
  --point 0.625,0.5208333333 \
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
output directory. For the two box scenarios:

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
directory. The bundled manifest assumes your upstream checkout is available at
`../../extcode/sam3_baseline` relative to this repo root.

## Interactive Replay Example

This is the current CLI-first replacement for the point-refinement flow from
`sam3_for_sam1_task_example.ipynb`. Instead of a widget loop, the Candle example
replays a deterministic JSON script of positive and negative clicks against the
same `truck.jpg` image used in that notebook.

Use the included replay manifest:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --interactive /home/dnorthover/extcode/sam3_baseline/assets/images/truck.jpg \
  --interactive-script candle-examples/examples/sam3/interactive_replay.example.json \
  --output-dir candle-examples/examples/sam3/output/interactive_replay
```

The included replay manifest uses the notebook click locations `(520,375)`,
`(500,375)`, and `(1125,625)` converted to normalized coordinates. Because the
CLI replay is incremental, later steps append those notebook clicks rather than
resetting the session between calls.

Each iteration writes its own subdirectory under the requested output directory:

- `step_000_*/base.png`: the original image for that refinement step
- `step_000_*/overlay.png`: prompt clicks plus the selected predicted box and mask
- `step_000_*/mask.png`: grayscale mask for the selected prediction
- `step_000_*/summary.json`: click history, labels, score, normalized box, and artifact paths
- `interactive_session.json`: session-level summary for the whole replay

If you also provide `--point` or `--box` with `--interactive`, those prompts are
run first as an `initial_prompt` iteration and later replay steps append more
clicks on top.

## CUDA Example

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
cargo run -p candle-examples --example sam3 --features cuda -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
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
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/truck.jpg \
  --prompt "truck" \
  --box 0.3125,0.6145833333,0.1527777778,0.2291666667 \
  --box-label 1 \
  --point 0.2888888889,0.3125 \
  --point-label 1 \
  --point 0.625,0.5208333333 \
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
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/output
```

If you source `/opt/intel/oneapi/mkl/2025.3/env/vars.sh` first, `LD_LIBRARY_PATH`
will already contain the MKL runtime directory. The inline CUDA override above
keeps that existing value by appending `:$LD_LIBRARY_PATH`.

## Parity Harness

Step 11 is implemented as a stage-by-stage parity harness. It compares the Rust
pipeline against a reference bundle exported from upstream PyTorch for one fixed
image/prompt pair. The examples below use the
`sam3_image_predictor_example.ipynb` single-positive-box case as the baseline
parity scenario.

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
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --box 0.41796875,0.6527777778,0.0859375,0.5 \
  --box-label 1 \
  --output-dir candle-examples/examples/sam3/reference
```

This writes:

- `candle-examples/examples/sam3/reference/reference.safetensors`
- `candle-examples/examples/sam3/reference/reference.json`
- `candle-examples/examples/sam3/reference/overlay.png`
- `candle-examples/examples/sam3/reference/prediction_overlay.png`
- `candle-examples/examples/sam3/reference/prediction_overlay_all_kept.png`
- `candle-examples/examples/sam3/reference/mask.png`

That is the true upstream parity mode. It uses the same exact square resize as
the Python SAM3 image processor.

### Export Interactive Replay Reference Bundle

`export_reference.py` also supports deterministic interactive replay export via
`--interactive-script`:

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/truck.jpg \
  --interactive-script candle-examples/examples/sam3/interactive_replay.example.json \
  --output-dir candle-examples/examples/sam3/reference_interactive
```

This writes the same `reference.safetensors` / `reference.json` pair, but the
metadata contains replay `steps` and the tensor bundle contains per-step:

- `step.N.geometry.features`
- `step.N.geometry.padding_mask`
- `step.N.fusion.memory`
- `step.N.decoder.pred_logits`
- `step.N.decoder.pred_boxes_xyxy`
- `step.N.segmentation.mask_logits`
- optional `step.N.decoder.presence_logits`

It also writes rendered artifacts for each replay step under:

- `step_000_<name>/base.png`
- `step_000_<name>/overlay.png`
- `step_000_<name>/prediction_overlay.png`
- `step_000_<name>/prediction_overlay_all_kept.png`
- `step_000_<name>/mask.png`
- `step_000_<name>/summary.json`

### Export `shoe` Text Reference Bundle

This matches the text-prompt part of `sam3_image_predictor_example.ipynb`.

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/reference_shoe
```

### Export Exact Positive/Negative Box Parity Bundles

These are the exact-resize geometry parity checks for the same
`sam3_image_predictor_example.ipynb` workflow. The positive case above is the
baseline example; this section provides the explicit positive/negative pair.

Positive box:

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --box 0.41796875,0.6527777778,0.0859375,0.5 \
  --box-label 1 \
  --output-dir candle-examples/examples/sam3/reference_box_positive
```

Negative box:

```bash
python candle-examples/examples/sam3/export_reference.py \
  --sam3-repo /home/dnorthover/extcode/sam3_baseline \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --image /home/dnorthover/extcode/sam3_baseline/assets/images/test_image.jpg \
  --box 0.41796875,0.6527777778,0.0859375,0.5 \
  --box-label 0 \
  --output-dir candle-examples/examples/sam3/reference_box_negative
```

### Run Rust Parity Check

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference \
  --output-dir candle-examples/examples/sam3/output
```

## Reference Comparison Mode

Reference bundle export and comparison are exact-only:

1. export the normal upstream exact bundle
2. run `--compare-reference-bundle` against that exact bundle
3. compare final boxes, scores, and masks in the generated report

Using the single-positive-box notebook case as the reference:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --compare-reference-bundle candle-examples/examples/sam3/reference \
  --output-dir candle-examples/examples/sam3/output/reference_compare
```

This writes:

- `candle-examples/examples/sam3/output/reference_compare/`
- `candle-examples/examples/sam3/output/reference_compare/reference_comparison_report.json`

The comparison report is output-level, not hidden-state parity. It includes:

- reference and Candle best query indices
- reference and Candle best scores
- box mean absolute difference
- box IoU
- mask mean absolute difference
- mask IoU at threshold `0.5`
- Candle `prediction_overlay.png` vs reference `prediction_overlay_all_kept.png` mean absolute difference
- Candle `prediction_overlay.png` vs reference `prediction_overlay_all_kept.png` RMSE

You can run the same comparison mode against the explicit positive/negative box
reference bundles too.

Positive box reference comparison:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --compare-reference-bundle candle-examples/examples/sam3/reference_box_positive \
  --output-dir candle-examples/examples/sam3/output/reference_compare_box_positive
```

Negative box reference comparison:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --compare-reference-bundle candle-examples/examples/sam3/reference_box_negative \
  --output-dir candle-examples/examples/sam3/output/reference_compare_box_negative
```

Each run writes render outputs directly into the requested output directory,
plus a top-level `reference_comparison_report.json` for that case.

You can do the same for the `shoe` text example:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --compare-reference-bundle candle-examples/examples/sam3/reference_shoe \
  --output-dir candle-examples/examples/sam3/output/reference_compare_shoe
```

Interactive replay bundles exported by `export_reference.py --interactive-script`
can be compared through the same flag. The CLI auto-detects the replay bundle
and runs the step-by-step interactive comparison across the full replay script
before returning failure:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --compare-reference-bundle candle-examples/examples/sam3/reference_interactive \
  --output-dir candle-examples/examples/sam3/output/reference_compare_interactive
```

That writes:

- `candle-examples/examples/sam3/output/reference_compare_interactive/interactive_comparison_report.json`
- `candle-examples/examples/sam3/output/reference_compare_interactive/step_000_<name>/...`
- `candle-examples/examples/sam3/output/reference_compare_interactive/step_001_<name>/...`
- `candle-examples/examples/sam3/output/reference_compare_interactive/step_002_<name>/...`

The interactive report includes per-step stage diffs plus final score / box /
mask comparison metrics for each replay step, and Candle also saves rendered
per-step masks and masked-image overlays in the output directory.

For the box-conditioned geometry checks above, run parity like this.

Exact positive:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference_box_positive \
  --output-dir candle-examples/examples/sam3/output/parity_box_positive \
  --cpu
```

Exact negative:

```bash
cargo run -p candle-examples --example sam3 --release -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3 \
  --parity-bundle candle-examples/examples/sam3/reference_box_negative \
  --output-dir candle-examples/examples/sam3/output/parity_box_negative \
  --cpu
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
- `--interactive` and `--video` are Phase 12 work-in-progress entry points, not full notebook parity yet.
