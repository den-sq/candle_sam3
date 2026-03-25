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

## Notes

- The example prints intermediate stage shapes so it still works as a debug harness.
- Geometry mask prompts are still not available because the current checkpoint does not
  include a geometry mask-encoder branch.
- Full video / tracker support is still outside this example.
