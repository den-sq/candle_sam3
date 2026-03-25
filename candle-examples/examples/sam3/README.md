# SAM 3 Example

This example is a smoke harness for the SAM 3 stages that are implemented so
far. It does not claim end-to-end grounding works yet, but it can exercise the
checkpoint loader, typed image state, text encoder, vision trunk, FPN neck, and
geometry encoder.

## Intended module split

- `sam3/config.rs`: upstream-shaped defaults and staging knobs
- `sam3/text.rs`: CLIP-like text encoder + resize projection
- `sam3/vitdet.rs`: ViTDet-style visual trunk
- `sam3/neck.rs`: simple-FPN neck and SAM2 side-neck hook
- `sam3/geometry.rs`: box/point/mask prompt encoding
- `sam3/encoder.rs`: visual-language fusion encoder
- `sam3/decoder.rs`: DETR-style decoder with presence token
- `sam3/segmentation.rs`: MaskFormer-like mask head
- `sam3/image.rs`: typed image-facing API for `set_image` and grounding

## Expected implementation order

1. Image-only grounding parity against upstream SAM 3 detector.
2. Checkpoint namespace mapping from `sam3.pt` into Candle `VarBuilder`.
3. Interactive image refinement using the SAM-style path.
4. Video detector/tracker integration.

## Checkpoint note

The example accepts either:

- a direct path to `sam3.pt`
- a Hugging Face repo directory containing `sam3.pt`

It also accepts either:

- a direct path to `tokenizer.json`
- a repo directory containing `tokenizer.json`

## Implemented smoke stages

- Text stage: tokenization plus the CLIP-like text encoder and resize projection
- Vision stage: image preprocessing, ViTDet trunk, and simple-FPN neck
- Geometry stage: empty-prompt encoding plus optional point/box prompt encoding

The geometry stage uses the currently loaded image checkpoint. That checkpoint
contains point and box prompt weights, but not a mask-prompt branch, so mask
prompts are intentionally not part of the example yet.

## Text encoder status

- `sam3/text.rs` runs the CLIP-like text transformer and resize projection
- tokenization stays in the example crate instead of `candle-transformers`
- the example pads and truncates to the SAM3 text context length before running
  the encoder

## Vision and geometry status

- `sam3/vitdet.rs` runs the checkpoint-backed image trunk
- `sam3/neck.rs` projects that output into FPN levels plus position encodings
- `sam3/geometry.rs` encodes:
  - an empty prompt via the CLS token path
  - optional `--point x,y` prompts
  - optional `--box cx,cy,w,h` prompts

Coordinates are normalized to `[0, 1]`.

For faster CPU smoke tests, pass `--smoke-image-size 336` to run the implemented
vision and geometry stages on a smaller square resize while keeping the model
configuration unchanged.

## Example commands

Text-only smoke:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --prompt "a white sneaker"
```

Vision plus empty-geometry smoke:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --smoke-image-size 336
```

Vision plus point/box geometry smoke:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --smoke-image-size 336 \
  --point 0.5,0.5 \
  --box 0.5,0.5,0.4,0.4
```

Combined text, vision, and geometry smoke:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/hf_sam3 \
  --tokenizer /path/to/hf_sam3 \
  --image candle-examples/examples/wuerstchen/assets/cat.jpg \
  --smoke-image-size 336 \
  --prompt "a cat" \
  --point 0.45,0.45 \
  --box 0.5,0.5,0.6,0.6
```
