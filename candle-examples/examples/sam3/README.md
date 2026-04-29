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
- upstream notebook-style example ports:
  - `sam3_image_predictor_example.ipynb`
  - `sam3_image_batched_inference.ipynb`
  - `sam3_video_predictor_example.ipynb`
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

## Checkpoint Setup

Source the helper below once per shell before running the checkpoint-backed
commands. It prompts for your Hugging Face username, Hugging Face access token,
and a download directory that defaults to `./chkpts`.

```bash
source candle-examples/examples/sam3/setup_sam3_env.sh
```

The helper downloads `sam3.pt` and `tokenizer.json` from the gated
`facebook/sam3` repo, exports `SAM3_CHECKPOINT_DIR`, `SAM3_CHECKPOINT`, and
`SAM3_TOKENIZER`, and writes a reusable `sam3-paths.env` file next to the
downloads.

## Example Commands

CPU image inference:

```bash
cargo run -p candle-examples --example sam3 -- \
  --image /path/to/image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/output
```

CUDA image inference:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --image /path/to/image.jpg \
  --prompt "shoe" \
  --output-dir candle-examples/examples/sam3/output
```

Geometry-only image inference:

```bash
cargo run -p candle-examples --example sam3 -- \
  --image /path/to/image.jpg \
  --point 0.29,0.31 \
  --point-label 1 \
  --output-dir candle-examples/examples/sam3/output
```

CUDA geometry-only image inference:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --image /path/to/image.jpg \
  --point 0.29,0.31 \
  --point-label 1 \
  --output-dir candle-examples/examples/sam3/output
```

Mixed text + geometry inference:

```bash
cargo run -p candle-examples --example sam3 -- \
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

CUDA mixed text + geometry inference:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
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
  --batch-manifest candle-examples/examples/sam3/batch_manifest.example.json \
  --output-dir candle-examples/examples/sam3/output/batch
```

CUDA batch-manifest execution:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --batch-manifest candle-examples/examples/sam3/batch_manifest.example.json \
  --output-dir candle-examples/examples/sam3/output/batch
```

Notebook example port execution:

```bash
cargo run -p candle-examples --example sam3 -- \
  --notebook-asset-root /path/to/upstream/sam3 \
  --notebook-example image-predictor \
  --output-dir candle-examples/examples/sam3/output
```

CUDA notebook example port execution:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --notebook-asset-root /path/to/upstream/sam3 \
  --notebook-example image-predictor \
  --output-dir candle-examples/examples/sam3/output
```

The notebook runners write under:

- `output/sam3_image_predictor_example/`
- `output/sam3_image_batched_inference/`
- `output/sam3_video_predictor_example/`

Interactive replay:

```bash
cargo run -p candle-examples --example sam3 -- \
  --interactive /path/to/image.jpg \
  --interactive-script candle-examples/examples/sam3/interactive_replay.example.json \
  --output-dir candle-examples/examples/sam3/output/interactive
```

CUDA interactive replay:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --interactive /path/to/image.jpg \
  --interactive-script candle-examples/examples/sam3/interactive_replay.example.json \
  --output-dir candle-examples/examples/sam3/output/interactive
```

Video prediction:

```bash
cargo run -p candle-examples --example sam3 -- \
  --video /path/to/video.mp4 \
  --video-prompt "person" \
  --output-dir candle-examples/examples/sam3/output/video
```

CUDA video prediction:

```bash
PATH="/usr/local/cuda-12.9/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.9 \
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" \
cargo run -p candle-examples --features cuda --example sam3 -- \
  --video /path/to/video.mp4 \
  --video-prompt "person" \
  --output-dir candle-examples/examples/sam3/output/video
```

## Paths

The example accepts either:

- a direct `sam3.pt` path, or a repo directory containing `sam3.pt`
- a direct `tokenizer.json` path, or a repo directory containing `tokenizer.json`
- `SAM3_CHECKPOINT` and `SAM3_TOKENIZER` environment variables when the flags
  are omitted

## Repo Split

Moved to `sam_parity`:

- upstream export scripts
- strict-port matrix generation
- Python-side parity/debug helpers
- image, interactive, and video parity comparisons
- fixture-backed parity docs and manifests

This repo keeps the SAM3 runtime implementation, example inference entrypoints,
and non-parity smoke/unit coverage.

## Notebook Notes

- `sam3_image_batched_inference` caches the two COCO notebook images under its
  output subdirectory before running the Rust port.
- `sam3_video_predictor_example` follows the upstream sequence as closely as
  the current Candle predictor API allows and records any multi-object
  incompatibility in the phase summaries.
- `sam3_agent.ipynb` still depends on an external multimodal LLM orchestration
  loop and is not implemented in the local Candle runtime yet.
