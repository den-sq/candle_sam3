#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args():
    def parse_box(value: str):
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"expected cx,cy,w,h for --box, got {value!r}"
            )
        return parts

    def parse_box_label(value: str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "pos", "positive"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "neg", "negative"}:
            return False
        raise argparse.ArgumentTypeError(
            f"expected boolean-ish box label for --box-label, got {value!r}"
        )

    parser = argparse.ArgumentParser(
        description="Export a SAM3 stage-by-stage parity bundle from upstream PyTorch."
    )
    parser.add_argument(
        "--sam3-repo",
        required=True,
        help="Path to the local facebookresearch/sam3 repository root.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to sam3.pt or a directory containing sam3.pt.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default=None, help="Optional text prompt to encode.")
    parser.add_argument(
        "--box",
        action="append",
        default=[],
        type=parse_box,
        help="Optional normalized box prompt in cx,cy,w,h format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--box-label",
        action="append",
        default=[],
        type=parse_box_label,
        help="Optional boolean-ish box label aligned with --box. Defaults to true for each box.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reference.safetensors and reference.json will be written.",
    )
    parser.add_argument(
        "--bpe-path",
        default=None,
        help="Optional path to bpe_simple_vocab_16e6.txt.gz. Defaults to <sam3-repo>/assets/.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1008,
        help="Square image size used by the upstream processor.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["exact", "crop_fill"],
        default="exact",
        help="Image preprocessing used for the exported bundle. `exact` matches upstream SAM3; `crop_fill` is an example-specific diagnostic mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit torch device, e.g. cpu or cuda. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Export only inputs, text, trunk, and FPN stages, skipping prompt/fusion/decoder/segmentation.",
    )
    parser.add_argument(
        "--debug-block",
        type=int,
        action="append",
        default=[],
        help="Export internal tensors for the specified ViT block index. Can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_repo_file(path: str, expected: str) -> Path:
    path = Path(path)
    return path / expected if path.is_dir() else path


def resolve_sam3_package_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "model_builder.py").exists():
        return path
    if (path / "sam3" / "model_builder.py").exists():
        return path / "sam3"
    raise FileNotFoundError(
        f"could not find sam3/model_builder.py under {path}; pass either the repo root or the inner sam3 package directory"
    )


def to_cpu_contiguous(tensor):
    return tensor.detach().to("cpu").contiguous()


def to_cpu_nchw(tensor):
    return to_cpu_contiguous(tensor.permute(0, 3, 1, 2))


def build_preprocessed_image(v2, image_tensor, image_size: int, preprocess_mode: str):
    if preprocess_mode == "exact":
        image = v2.functional.resize(
            image_tensor,
            [image_size, image_size],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        image = v2.functional.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image.unsqueeze(0), None

    orig_h, orig_w = image_tensor.shape[-2:]
    scale = max(image_size / orig_h, image_size / orig_w)
    resized_h = round(orig_h * scale)
    resized_w = round(orig_w * scale)
    crop_y = max(0, (resized_h - image_size) // 2)
    crop_x = max(0, (resized_w - image_size) // 2)
    image = v2.functional.resize(
        image_tensor,
        [resized_h, resized_w],
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True,
    )
    image = v2.functional.crop(image, crop_y, crop_x, image_size, image_size)
    image = v2.functional.to_dtype(image, torch.float32, scale=True)
    image = v2.functional.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = {
        "resized_h": resized_h,
        "resized_w": resized_w,
        "crop_y": crop_y,
        "crop_x": crop_x,
        "image_size": image_size,
        "orig_h": orig_h,
        "orig_w": orig_w,
    }
    return image.unsqueeze(0), transform


def map_box_to_crop_fill(box, transform):
    cx, cy, w, h = box
    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    x1 = cx + w * 0.5
    y1 = cy + h * 0.5
    x0 = ((x0 * transform["resized_w"]) - transform["crop_x"]) / transform["image_size"]
    y0 = ((y0 * transform["resized_h"]) - transform["crop_y"]) / transform["image_size"]
    x1 = ((x1 * transform["resized_w"]) - transform["crop_x"]) / transform["image_size"]
    y1 = ((y1 * transform["resized_h"]) - transform["crop_y"]) / transform["image_size"]
    return [(x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0]


def map_box_from_crop_fill(box_xyxy, transform):
    x0 = (transform["crop_x"] + box_xyxy[0] * transform["image_size"]) / transform["resized_w"]
    y0 = (transform["crop_y"] + box_xyxy[1] * transform["image_size"]) / transform["resized_h"]
    x1 = (transform["crop_x"] + box_xyxy[2] * transform["image_size"]) / transform["resized_w"]
    y1 = (transform["crop_y"] + box_xyxy[3] * transform["image_size"]) / transform["resized_h"]
    return [x0, y0, x1, y1]


def normalized_box_to_pixels(box_xyxy, width, height):
    x0 = round(max(0.0, min(1.0, box_xyxy[0])) * max(width - 1, 0))
    y0 = round(max(0.0, min(1.0, box_xyxy[1])) * max(height - 1, 0))
    x1 = round(max(0.0, min(1.0, box_xyxy[2])) * max(width - 1, 0))
    y1 = round(max(0.0, min(1.0, box_xyxy[3])) * max(height - 1, 0))
    return [x0, y0, x1, y1]


def draw_prompt_annotations(image, boxes, box_labels):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box, label in zip(boxes, box_labels):
        x0, y0, x1, y1 = normalized_box_to_pixels(
            [box[0] - box[2] * 0.5, box[1] - box[3] * 0.5, box[0] + box[2] * 0.5, box[1] + box[3] * 0.5],
            width,
            height,
        )
        color = (59, 130, 246, 255) if label else (239, 68, 68, 255)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)


def palette_color(index):
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    return palette[index % len(palette)]


def best_kept_query(scores, threshold=0.5):
    best_any_idx = 0
    best_any_score = float("-inf")
    best_kept = None
    flat = scores[0, :, 0].detach().cpu()
    for idx, score in enumerate(flat.tolist()):
        if score > best_any_score:
            best_any_idx = idx
            best_any_score = score
        if score > threshold and (best_kept is None or score > best_kept[1]):
            best_kept = (idx, score)
    return best_kept if best_kept is not None else (best_any_idx, best_any_score)


def restore_crop_fill_mask(mask_probs, transform):
    from PIL import Image

    target_size = transform["image_size"]
    mask_image = Image.fromarray((mask_probs.clip(0.0, 1.0) * 255.0).round().astype("uint8"), mode="L")
    canvas = Image.new("L", (transform["resized_w"], transform["resized_h"]), 0)
    canvas.paste(mask_image, (transform["crop_x"], transform["crop_y"]))
    return canvas.resize((transform["orig_w"], transform["orig_h"]), Image.Resampling.BILINEAR)


def upsample_mask_to_original(mask_logits, image_size, image_size_hw, crop_fill_transform):
    from PIL import Image
    import torch.nn.functional as F

    orig_h, orig_w = image_size_hw
    probs = torch.sigmoid(
        F.interpolate(
            mask_logits.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    ).detach().cpu().numpy()
    if crop_fill_transform is not None:
        restored = restore_crop_fill_mask(probs, crop_fill_transform)
        return restored, probs
    image = Image.fromarray((probs.clip(0.0, 1.0) * 255.0).round().astype("uint8"), mode="L")
    return image.resize((orig_w, orig_h), Image.Resampling.BILINEAR), probs


def blend_mask_on_image(image, mask_image, color=(56, 201, 84), threshold=0.5):
    import numpy as np
    from PIL import Image

    rgba = np.array(image.convert("RGBA"), dtype=np.float32)
    mask = np.array(mask_image, dtype=np.float32) / 255.0
    on = mask >= threshold
    alpha = 0.35
    for channel, value in enumerate(color):
        rgba[..., channel] = np.where(
            on,
            (1.0 - alpha) * rgba[..., channel] + alpha * float(value),
            rgba[..., channel],
        )
    rgba[..., 3] = 255.0
    return Image.fromarray(rgba.clip(0, 255).astype("uint8"), mode="RGBA")


def draw_prediction_box(image, box_xyxy, color, score=None, index=None):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    x0, y0, x1, y1 = normalized_box_to_pixels(box_xyxy, width, height)
    draw.rectangle([x0, y0, x1, y1], outline=tuple(color) + (255,), width=3)
    label_parts = []
    if index is not None:
        label_parts.append(f"id={index}")
    if score is not None:
        label_parts.append(f"{score:.2f}")
    if label_parts:
        draw.text((x0, max(0, y0 - 12)), ", ".join(label_parts), fill=tuple(color) + (255,))


def main():
    args = parse_args()
    if args.box_label and len(args.box_label) != len(args.box):
        raise ValueError(
            f"--box-label count ({len(args.box_label)}) must match --box count ({len(args.box)})"
        )
    if args.prompt is None and not args.box:
        raise ValueError("provide --prompt, --box, or both")
    effective_prompt = args.prompt
    if effective_prompt is None and args.box:
        effective_prompt = "visual"

    sam3_package_dir = resolve_sam3_package_dir(Path(args.sam3_repo))
    sys.path.insert(0, str(sam3_package_dir.parent))

    from PIL import Image
    from safetensors.torch import save_file
    import torchvision
    from torchvision.transforms import v2

    import sam3.model_builder as sam3_model_builder
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.geometry_encoders import SequenceGeometryEncoder
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.vitdet import get_abs_pos, window_partition, window_unpartition

    checkpoint_path = resolve_repo_file(args.checkpoint, "sam3.pt").expanduser().resolve()
    bpe_path = (
        Path(args.bpe_path).expanduser().resolve()
        if args.bpe_path is not None
        else sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_boxes = list(args.box)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        def create_cpu_position_encoding(precompute_resolution=None):
            return PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000,
                precompute_resolution=None,
            )

        sam3_model_builder._create_position_encoding = create_cpu_position_encoding

        def get_coords_cpu_safe(H, W, device):
            if device == "cuda":
                device = "cpu"
            coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
            coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
            return coords_h, coords_w

        TransformerDecoder._get_coords = staticmethod(get_coords_cpu_safe)

        def encode_boxes_cpu_safe(self, boxes, boxes_mask, boxes_labels, img_feats):
            boxes_embed = None
            n_boxes, bs = boxes.shape[:2]

            if self.boxes_direct_project is not None:
                proj = self.boxes_direct_project(boxes)
                assert boxes_embed is None
                boxes_embed = proj

            if self.boxes_pool_project is not None:
                H, W = img_feats.shape[-2:]
                boxes_xyxy = box_cxcywh_to_xyxy(boxes)
                scale = torch.tensor(
                    [W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
                ).view(1, 1, 4)
                boxes_xyxy = boxes_xyxy * scale
                sampled = torchvision.ops.roi_align(
                    img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
                )
                assert list(sampled.shape) == [
                    bs * n_boxes,
                    self.d_model,
                    self.roi_size,
                    self.roi_size,
                ]
                proj = self.boxes_pool_project(sampled)
                proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
                if boxes_embed is None:
                    boxes_embed = proj
                else:
                    boxes_embed = boxes_embed + proj

            if self.boxes_pos_enc_project is not None:
                cx, cy, w, h = boxes.unbind(-1)
                enc = self.pos_enc.encode_boxes(
                    cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
                )
                enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

                proj = self.boxes_pos_enc_project(enc)
                if boxes_embed is None:
                    boxes_embed = proj
                else:
                    boxes_embed = boxes_embed + proj

            type_embed = self.label_embed(boxes_labels.long())
            return type_embed + boxes_embed, boxes_mask

        SequenceGeometryEncoder._encode_boxes = encode_boxes_cpu_safe

    def run_trunk_with_debug(trunk, image_tensor, debug_blocks):
        debug_blocks = set(debug_blocks)
        x = trunk.patch_embed(image_tensor)
        h, w = x.shape[1], x.shape[2]

        if trunk.pos_embed is not None:
            x = x + get_abs_pos(
                trunk.pos_embed,
                trunk.pretrain_use_cls_token,
                (h, w),
                trunk.retain_cls_token,
                tiling=trunk.tile_abs_pos,
            )

        x = trunk.ln_pre(x)

        if trunk.retain_cls_token:
            raise NotImplementedError("debug export does not support retained cls token")

        block_outputs = []
        debug_tensors = {}
        for block_idx, block in enumerate(trunk.blocks):
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.input"] = to_cpu_nchw(x)

            shortcut = x
            x_norm1 = block.norm1(x)
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.norm1"] = to_cpu_nchw(x_norm1)

            if block.window_size > 0:
                hw = (x_norm1.shape[1], x_norm1.shape[2])
                x_attn, pad_hw = window_partition(x_norm1, block.window_size)
            else:
                hw = None
                x_attn = x_norm1

            x_attn = block.ls1(block.attn(x_attn))
            if block.window_size > 0:
                x_attn = window_unpartition(x_attn, block.window_size, pad_hw, hw)
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.attn_output"] = to_cpu_nchw(x_attn)

            x = shortcut + block.dropout(block.drop_path(x_attn))
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.post_attn"] = to_cpu_nchw(x)

            x_norm2 = block.norm2(x)
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.norm2"] = to_cpu_nchw(x_norm2)

            mlp_fc1 = block.mlp.fc1(x_norm2)
            mlp_gelu = block.mlp.act(mlp_fc1)
            mlp_fc2 = block.mlp.fc2(mlp_gelu)
            mlp_output = block.dropout(block.drop_path(block.ls2(mlp_fc2)))
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.mlp_fc1"] = to_cpu_nchw(
                    mlp_fc1
                )
                debug_tensors[f"vision.block_debug.{block_idx}.mlp_gelu"] = to_cpu_nchw(
                    mlp_gelu
                )
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.mlp_output"] = to_cpu_nchw(
                    mlp_output
                )

            x = x + mlp_output
            if block_idx in debug_blocks:
                debug_tensors[f"vision.block_debug.{block_idx}.output"] = to_cpu_nchw(x)

            block_outputs.append((block_idx, to_cpu_nchw(x)))

        return [to_cpu_contiguous(x.permute(0, 3, 1, 2))], block_outputs, debug_tensors

    image = Image.open(args.image).convert("RGB")
    model = sam3_model_builder.build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        device=str(device),
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    # Upstream SAM3 seeds the decoder RPB coordinate cache on hard-coded CUDA
    # when resolution/stride are configured, which breaks CPU-only parity export.
    decoder = model.transformer.decoder
    if hasattr(decoder, "compilable_cord_cache"):
        decoder.compilable_cord_cache = None
        decoder.compilable_stored_size = None

    processor = Sam3Processor(
        model=model,
        resolution=args.image_size,
        device=str(device),
        confidence_threshold=0.5,
    )

    with torch.inference_mode():
        image_tensor = v2.functional.to_image(image).to(device)
        preprocessed_image, crop_fill_transform = build_preprocessed_image(
            v2, image_tensor, args.image_size, args.preprocess_mode
        )
        if args.box:
            exported_boxes = [
                map_box_to_crop_fill(box, crop_fill_transform)
                if crop_fill_transform is not None
                else box
                for box in args.box
            ]
        trunk_outputs, block_outputs, debug_tensors = run_trunk_with_debug(
            model.backbone.vision_backbone.trunk,
            preprocessed_image,
            args.debug_block,
        )

        state = {
            "original_height": image.height,
            "original_width": image.width,
            "backbone_out": model.backbone.forward_image(preprocessed_image),
        }
        find_input = processor.find_stage

        tokenizer = model.backbone.language_backbone.tokenizer
        context_length = model.backbone.language_backbone.context_length
        input_ids = tokenizer([effective_prompt], context_length=context_length).to(device)
        attention_mask = (input_ids != 0).to(torch.uint8)

        text_outputs = model.backbone.forward_text([effective_prompt], device=device)
        state["backbone_out"].update(text_outputs)
        backbone_out = state["backbone_out"]
        if not args.vision_only:
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = model._get_dummy_prompt()
            if args.box:
                box_labels = (
                    args.box_label if args.box_label else [True for _ in args.box]
                )
                for model_box, label in zip(exported_boxes, box_labels):
                    boxes = torch.tensor(
                        model_box, device=device, dtype=torch.float32
                    ).view(1, 1, 4)
                    labels = torch.tensor([label], device=device, dtype=torch.bool).view(
                        1, 1
                    )
                    state["geometric_prompt"].append_boxes(boxes, labels)

            prompt, prompt_mask, backbone_out = model._encode_prompt(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=state["geometric_prompt"],
            )
            backbone_out, encoder_out, _ = model._run_encoder(
                backbone_out=backbone_out,
                find_input=find_input,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
            out, hs = model._run_decoder(
                pos_embed=encoder_out["pos_embed"],
                memory=out["encoder_hidden_states"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

    tensors = {
        "inputs.image": to_cpu_contiguous(preprocessed_image),
        "inputs.input_ids": to_cpu_contiguous(input_ids),
        "inputs.attention_mask": to_cpu_contiguous(attention_mask),
        "text.input_embeddings": to_cpu_contiguous(text_outputs["language_embeds"]),
        "text.memory": to_cpu_contiguous(text_outputs["language_features"]),
    }
    if args.box:
        box_label_tensor = torch.tensor(
            args.box_label if args.box_label else [True for _ in args.box],
            dtype=torch.uint8,
        )
        tensors["inputs.boxes_cxcywh"] = to_cpu_contiguous(
            torch.tensor(exported_boxes, dtype=torch.float32)
        )
        tensors["inputs.box_labels"] = to_cpu_contiguous(box_label_tensor)
    if not args.vision_only:
        tensors.update(
            {
                "fusion.memory": to_cpu_contiguous(encoder_out["encoder_hidden_states"]),
                "geometry.features": to_cpu_contiguous(prompt[text_outputs["language_features"].shape[0] :]),
                "geometry.padding_mask": to_cpu_contiguous(
                    prompt_mask[:, text_outputs["language_mask"].shape[1] :].to(torch.uint8)
                ),
                "decoder.pred_logits": to_cpu_contiguous(out["pred_logits"]),
                "decoder.pred_boxes_xyxy": to_cpu_contiguous(out["pred_boxes_xyxy"]),
                "segmentation.mask_logits": to_cpu_contiguous(out["pred_masks"]),
            }
        )
        if encoder_out.get("pos_embed") is not None:
            tensors["fusion.pos_embed"] = to_cpu_contiguous(encoder_out["pos_embed"])
        if encoder_out.get("padding_mask") is not None:
            tensors["fusion.padding_mask"] = to_cpu_contiguous(
                encoder_out["padding_mask"].to(torch.uint8)
            )
        if encoder_out.get("spatial_shapes") is not None:
            tensors["fusion.spatial_shapes"] = to_cpu_contiguous(encoder_out["spatial_shapes"])
        if encoder_out.get("level_start_index") is not None:
            tensors["fusion.level_start_index"] = to_cpu_contiguous(
                encoder_out["level_start_index"]
            )
        if encoder_out.get("valid_ratios") is not None:
            tensors["fusion.valid_ratios"] = to_cpu_contiguous(encoder_out["valid_ratios"])
        if "presence_logit_dec" in out:
            tensors["decoder.presence_logits"] = to_cpu_contiguous(out["presence_logit_dec"])
        if "semantic_logits" in out:
            tensors["segmentation.semantic_logits"] = to_cpu_contiguous(out["semantic_logits"])
        if "presence_logits" in out:
            tensors["segmentation.presence_logits"] = to_cpu_contiguous(out["presence_logits"])

    for level_idx, feature_map in enumerate(backbone_out["backbone_fpn"]):
        tensors[f"vision.backbone_fpn.{level_idx}"] = to_cpu_contiguous(feature_map)
    for block_idx, feature_map in block_outputs:
        tensors[f"vision.block.{block_idx}"] = feature_map
    for name, feature_map in debug_tensors.items():
        tensors[name] = feature_map
    for level_idx, feature_map in enumerate(trunk_outputs):
        tensors[f"vision.trunk.{level_idx}"] = to_cpu_contiguous(feature_map)

    debug_stage_order = []
    for block_idx in args.debug_block:
        for suffix in [
            "input",
            "norm1",
            "attn_output",
            "post_attn",
            "norm2",
            "mlp_fc1",
            "mlp_gelu",
            "mlp_output",
            "output",
        ]:
            name = f"vision.block_debug.{block_idx}.{suffix}"
            if name in tensors:
                debug_stage_order.append(name)

    stage_order = [
        "text.input_embeddings",
        "text.memory",
        *debug_stage_order,
        *[f"vision.block.{idx}" for idx, _ in block_outputs],
        *[f"vision.trunk.{idx}" for idx in range(len(trunk_outputs))],
        *[f"vision.backbone_fpn.{idx}" for idx in range(len(backbone_out["backbone_fpn"]))],
    ]
    if not args.vision_only:
        stage_order.extend(
            [
                "geometry.features",
                "geometry.padding_mask",
                "fusion.memory",
                "decoder.pred_logits",
                "decoder.pred_boxes_xyxy",
                "segmentation.mask_logits",
            ]
        )
        for optional_stage in [
            "fusion.pos_embed",
            "fusion.padding_mask",
            "fusion.spatial_shapes",
            "fusion.level_start_index",
            "fusion.valid_ratios",
            "decoder.presence_logits",
            "segmentation.semantic_logits",
            "segmentation.presence_logits",
        ]:
            if optional_stage in tensors:
                stage_order.append(optional_stage)

    if not args.vision_only:
        score_tensor = torch.sigmoid(out["pred_logits"])
        if "presence_logit_dec" in out:
            presence_scores = torch.sigmoid(out["presence_logit_dec"]).view(-1, 1, 1)
            score_tensor = score_tensor * presence_scores
        best_idx, best_score = best_kept_query(score_tensor, threshold=0.5)
        kept_scores = score_tensor[0, :, 0].detach().cpu()
        kept_indices = (kept_scores > 0.5).nonzero(as_tuple=False).flatten().tolist()

        prediction_box = out["pred_boxes_xyxy"][0, best_idx].detach().cpu().tolist()
        if crop_fill_transform is not None:
            prediction_box = map_box_from_crop_fill(prediction_box, crop_fill_transform)

        restored_mask, _raw_mask_probs = upsample_mask_to_original(
            out["pred_masks"][0, best_idx],
            args.image_size,
            (image.height, image.width),
            crop_fill_transform,
        )
        prediction_overlay = image.convert("RGBA")
        prediction_overlay = blend_mask_on_image(prediction_overlay, restored_mask)

        from PIL import ImageDraw

        draw = ImageDraw.Draw(prediction_overlay)
        draw.rectangle(
            normalized_box_to_pixels(prediction_box, image.width, image.height),
            outline=(56, 201, 84, 255),
            width=3,
        )

        overlay = prediction_overlay.copy()
        draw_prompt_annotations(
            overlay,
            args.box,
            args.box_label if args.box_label else [True for _ in args.box],
        )

        all_kept_overlay = image.convert("RGBA")
        for kept_rank, kept_idx in enumerate(kept_indices):
            kept_box = out["pred_boxes_xyxy"][0, kept_idx].detach().cpu().tolist()
            if crop_fill_transform is not None:
                kept_box = map_box_from_crop_fill(kept_box, crop_fill_transform)
            kept_mask, _ = upsample_mask_to_original(
                out["pred_masks"][0, kept_idx],
                args.image_size,
                (image.height, image.width),
                crop_fill_transform,
            )
            color = palette_color(kept_rank)
            all_kept_overlay = blend_mask_on_image(all_kept_overlay, kept_mask, color=color)
            draw_prediction_box(
                all_kept_overlay,
                kept_box,
                color,
                score=float(kept_scores[kept_idx].item()),
                index=kept_rank,
            )

        restored_mask.save(output_dir / "mask.png")
        overlay.save(output_dir / "overlay.png")
        prediction_overlay.save(output_dir / "prediction_overlay.png")
        all_kept_overlay.save(output_dir / "prediction_overlay_all_kept.png")

    save_file(
        tensors,
        str(output_dir / "reference.safetensors"),
        metadata={"bundle_version": "1"},
    )
    metadata = {
        "bundle_version": 1,
        "image_path": str(Path(args.image).expanduser().resolve()),
        "prompt": args.prompt,
        "effective_prompt": effective_prompt,
        "boxes_cxcywh": args.box,
        "model_boxes_cxcywh": exported_boxes if args.box else [],
        "box_labels": args.box_label if args.box_label else [True for _ in args.box],
        "image_size": args.image_size,
        "preprocess_mode": args.preprocess_mode,
        "checkpoint_path": str(checkpoint_path),
        "bpe_path": str(bpe_path),
        "stage_order": stage_order,
    }
    with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved parity bundle to {output_dir}")
    print(f"  tensors: {output_dir / 'reference.safetensors'}")
    print(f"  metadata: {output_dir / 'reference.json'}")
    if not args.vision_only:
        print(f"  overlay: {output_dir / 'overlay.png'}")
        print(f"  prediction overlay: {output_dir / 'prediction_overlay.png'}")
        print(f"  all-kept overlay: {output_dir / 'prediction_overlay_all_kept.png'}")
        print(f"  mask: {output_dir / 'mask.png'}")


if __name__ == "__main__":
    main()
