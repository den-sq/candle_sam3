# SAM 3 Scaffold

This example mirrors the planned Candle integration layout for SAM 3 without
claiming the model is already ported.

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

The scaffold example accepts the upstream `sam3.pt` checkpoint and opens the
`model` state dictionary namespace, matching the current upstream checkpoint
layout.

## Text encoder status

Step 4 of the port plan is now implemented:

- `sam3/text.rs` runs the CLIP-like text transformer and resize projection.
- Tokenization stays in the example crate instead of `candle-transformers`.
- The example accepts `--tokenizer <tokenizer.json>` and pads/truncates to the
  SAM3 text context length before running the encoder.

Example:

```bash
cargo run -p candle-examples --example sam3 -- \
  --checkpoint /path/to/sam3.pt \
  --tokenizer /path/to/tokenizer.json \
  --prompt "a white sneaker"
```
