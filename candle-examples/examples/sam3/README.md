# SAM 3 Example

This example is now runtime-only. It covers local SAM3 inference flows in the
main Candle repo and no longer owns upstream export, parity comparison, or
fixture-generation tooling.

For parity/export workflows, use the sibling repo at
`/home/dnorthover/ChengCode/sam_parity`.

## Supported Runtime Flows

- text-guided image prediction
- geometry-only image prediction with points and boxes
- mixed text + geometry prompts
- sequential image batch manifests
- canned image-predictor replay mode
- interactive image refinement replay
- video prediction with rendered frame and mask outputs

The rendered image outputs remain:

- `overlay.png`
- `prediction_overlay.png`
- `prediction_overlay_all_kept.png`
- `mask.png`
- `mask_sigmoid.png`
- `mask_one_minus_sigmoid.png`
- `mask_sigmoid_threshold_0_5.png`
- `mask_one_minus_sigmoid_threshold_0_5.png`
- `prediction_overlay_sigmoid_threshold_0_5.png`
- `prediction_overlay_one_minus_sigmoid_threshold_0_5.png`
- `summary.json`

## Example Commands

CPU image inference:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /path/to/image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/output
```

Geometry-only image inference:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --image /path/to/image.jpg \
  --point 0.29,0.31 \
  --point-label 1 \
  --output-dir candle-examples/examples/sam3/output
```

Mixed text + geometry inference:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image /path/to/image.jpg \
  --prompt "truck" \
  --box 0.31,0.61,0.15,0.23 \
  --box-label 1 \
  --point 0.29,0.31 \
  --point-label 1 \
  --point 0.62,0.52 \
  --point-label 0 \
  --output-dir candle-examples/examples/sam3/output
```

Batch-manifest execution:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --batch-manifest candle-examples/examples/sam3/batch_manifest.example.json \
  --output-dir candle-examples/examples/sam3/output/batch
```

Interactive replay:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --interactive /path/to/image.jpg \
  --interactive-script candle-examples/examples/sam3/interactive_replay.example.json \
  --output-dir candle-examples/examples/sam3/output/interactive
```

Video prediction:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --video /path/to/video.mp4 \
  --video-prompt "person" \
  --output-dir candle-examples/examples/sam3/output/video
```

## Paths

The example accepts either:

- a direct `sam3.pt` path, or a repo directory containing `sam3.pt`
- a direct `tokenizer.json` path, or a repo directory containing `tokenizer.json`

## Repo Split

Moved to `sam_parity`:

- upstream export scripts
- strict-port matrix generation
- Python-side parity/debug helpers
- image, interactive, and video parity comparisons
- fixture-backed parity docs and manifests

This repo keeps the SAM3 runtime implementation, example inference entrypoints,
and non-parity smoke/unit coverage.
