#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
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
        description="Export SAM3 image parity bundles or video reference bundles from upstream PyTorch."
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
    parser.add_argument("--image", default=None, help="Input image path.")
    parser.add_argument(
        "--video",
        default=None,
        help="Optional input video path or extracted-frame directory for video reference export.",
    )
    parser.add_argument("--prompt", default=None, help="Optional text prompt to encode.")
    parser.add_argument(
        "--interactive-script",
        default=None,
        help="Optional JSON replay manifest with interactive point clicks to export step-by-step interactive reference outputs.",
    )
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
        help="Directory where the exported bundle artifacts will be written.",
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
        "--video-frame-count",
        type=int,
        default=30,
        help="Maximum number of video frames to export for video reference bundles.",
    )
    parser.add_argument(
        "--video-apply-temporal-disambiguation",
        action="store_true",
        help="Enable upstream temporal disambiguation for video export. Disabled by default so example bundles keep raw propagated masks instead of suppressing unconfirmed tracklets.",
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


def build_preprocessed_image(v2, image_tensor, image_size: int):
    image = v2.functional.resize(
        image_tensor,
        [image_size, image_size],
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True,
    )
    image = v2.functional.to_dtype(image, torch.float32, scale=True)
    image = v2.functional.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return image.unsqueeze(0)


def default_positive_label():
    return 1


def load_interactive_script(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        steps = raw
    else:
        steps = raw.get("steps", [])
    if not steps:
        raise ValueError(f"interactive replay script {path} does not contain any steps")

    parsed_steps = []
    accumulated_points = []
    accumulated_labels = []
    for idx, step in enumerate(steps):
        points = step.get("points", [])
        if not points:
            raise ValueError(f"interactive replay step {idx} does not contain any points")
        step_points = []
        step_labels = []
        for point in points:
            step_points.append([float(point["x"]), float(point["y"])])
            step_labels.append(int(point.get("label", default_positive_label())))
        accumulated_points.extend(step_points)
        accumulated_labels.extend(step_labels)
        parsed_steps.append(
            {
                "name": step.get("name"),
                "step_points_xy_normalized": step_points,
                "step_point_labels": step_labels,
                "accumulated_points_xy_normalized": [list(point) for point in accumulated_points],
                "accumulated_point_labels": list(accumulated_labels),
            }
        )
    return parsed_steps


def normalized_box_to_pixels(box_xyxy, width, height):
    x0 = round(max(0.0, min(1.0, box_xyxy[0])) * max(width - 1, 0))
    y0 = round(max(0.0, min(1.0, box_xyxy[1])) * max(height - 1, 0))
    x1 = round(max(0.0, min(1.0, box_xyxy[2])) * max(width - 1, 0))
    y1 = round(max(0.0, min(1.0, box_xyxy[3])) * max(height - 1, 0))
    return [x0, y0, x1, y1]


def prompt_color(label):
    return (59, 130, 246, 255) if label else (239, 68, 68, 255)


def draw_prompt_annotations(
    image,
    boxes=None,
    box_labels=None,
    points=None,
    point_labels=None,
):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    boxes = boxes or []
    box_labels = box_labels or []
    points = points or []
    point_labels = point_labels or []
    for box, label in zip(boxes, box_labels):
        x0, y0, x1, y1 = normalized_box_to_pixels(
            [box[0] - box[2] * 0.5, box[1] - box[3] * 0.5, box[0] + box[2] * 0.5, box[1] + box[3] * 0.5],
            width,
            height,
        )
        color = prompt_color(label)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    radius = 5
    for point, label in zip(points, point_labels):
        px = round(max(0.0, min(1.0, point[0])) * max(width - 1, 0))
        py = round(max(0.0, min(1.0, point[1])) * max(height - 1, 0))
        color = prompt_color(label)
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=color,
            outline=color,
        )


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


def upsample_mask_to_original(mask_logits, image_size, image_size_hw):
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


def sanitize_step_name(step_name):
    sanitized = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in step_name
    ).strip("_")
    return sanitized or "step"


def interactive_step_dir(output_dir, step_idx, step_name):
    return output_dir / f"step_{step_idx:03d}_{sanitize_step_name(step_name)}"


def render_interactive_reference_step(
    image,
    output_dir,
    step_idx,
    step_name,
    script_step_index,
    image_size,
    pred_logits,
    pred_boxes_xyxy,
    pred_masks,
    accumulated_points,
    accumulated_point_labels,
):
    step_dir = interactive_step_dir(output_dir, step_idx, step_name)
    step_dir.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGBA")
    base_path = step_dir / "base.png"
    base.save(base_path)

    score_tensor = torch.sigmoid(pred_logits)
    best_idx, best_score = best_kept_query(score_tensor, threshold=0.5)
    kept_scores = score_tensor[0, :, 0].detach().cpu()
    kept_indices = (kept_scores > 0.5).nonzero(as_tuple=False).flatten().tolist()
    best_box = pred_boxes_xyxy[0, best_idx].detach().cpu().tolist()

    restored_mask, raw_mask_probs = upsample_mask_to_original(
        pred_masks[0, best_idx], image_size, (image.height, image.width)
    )
    prediction_overlay = blend_mask_on_image(base.copy(), restored_mask)
    draw_prediction_box(
        prediction_overlay,
        best_box,
        (56, 201, 84),
        score=float(best_score),
        index=int(best_idx),
    )

    overlay = prediction_overlay.copy()
    draw_prompt_annotations(
        overlay,
        points=accumulated_points,
        point_labels=accumulated_point_labels,
    )

    all_kept_overlay = base.copy()
    kept_queries_debug = []
    for rank, kept_idx in enumerate(kept_indices):
        kept_box = pred_boxes_xyxy[0, kept_idx].detach().cpu().tolist()
        kept_mask, _ = upsample_mask_to_original(
            pred_masks[0, kept_idx], image_size, (image.height, image.width)
        )
        color = palette_color(rank)
        all_kept_overlay = blend_mask_on_image(all_kept_overlay, kept_mask, color=color)
        draw_prediction_box(
            all_kept_overlay,
            kept_box,
            color,
            score=float(kept_scores[kept_idx].item()),
            index=rank,
        )
        kept_queries_debug.append(
            {
                "rank": rank,
                "query_index": int(kept_idx),
                "score": float(kept_scores[kept_idx].item()),
                "box_xyxy_normalized": kept_box,
            }
        )

    mask_path = step_dir / "mask.png"
    overlay_path = step_dir / "overlay.png"
    prediction_overlay_path = step_dir / "prediction_overlay.png"
    prediction_overlay_all_kept_path = step_dir / "prediction_overlay_all_kept.png"
    restored_mask.save(mask_path)
    overlay.save(overlay_path)
    prediction_overlay.save(prediction_overlay_path)
    all_kept_overlay.save(prediction_overlay_all_kept_path)

    summary = {
        "iteration_index": step_idx,
        "step_name": step_name,
        "script_step_index": script_step_index,
        "best_query_index": int(best_idx),
        "best_score": float(best_score),
        "best_box_xyxy_normalized": best_box,
        "accumulated_points_xy_normalized": accumulated_points,
        "accumulated_point_labels": accumulated_point_labels,
        "render_image_size": {"width": image.width, "height": image.height},
        "base_path": str(base_path),
        "overlay_path": str(overlay_path),
        "prediction_overlay_path": str(prediction_overlay_path),
        "prediction_overlay_all_kept_path": str(prediction_overlay_all_kept_path),
        "mask_path": str(mask_path),
        "kept_queries_debug": kept_queries_debug,
        "mask_mean_probability": float(raw_mask_probs.mean()),
    }
    (step_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_frame_names(path: Path):
    stem = path.stem
    return (0, int(stem), path.name.lower()) if stem.isdigit() else (1, stem.lower(), path.name.lower())


def sorted_frame_paths(dir_path: Path):
    frame_paths = [
        path
        for path in dir_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ]
    frame_paths.sort(key=compare_frame_names)
    if not frame_paths:
        raise ValueError(f"no image frames found in {dir_path}")
    return frame_paths


def resolve_tokenizer_path(checkpoint_path: Path):
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "tokenizer.json"
        if candidate.exists():
            return candidate.resolve()
    else:
        candidate = checkpoint_path.parent / "tokenizer.json"
        if candidate.exists():
            return candidate.resolve()
    return None


def prepare_video_frames(video_path: Path, frames_dir: Path, max_frames: int):
    from PIL import Image

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if video_path.is_dir():
        source_paths = sorted_frame_paths(video_path)[:max_frames]
        for frame_idx, source_path in enumerate(source_paths):
            frame = Image.open(source_path).convert("RGB")
            frame.save(frames_dir / f"{frame_idx:06d}.png")
        return sorted_frame_paths(frames_dir)

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        str(max_frames),
        "-start_number",
        "0",
        str(frames_dir / "%06d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while extracting frames from {video_path}: {result.stderr.strip()}"
        )
    frame_paths = sorted_frame_paths(frames_dir)
    if len(frame_paths) > max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return frame_paths


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def box_cxcywh_to_xywh(box):
    x0, y0, x1, y1 = box_cxcywh_to_xyxy(box)
    return [x0, y0, x1 - x0, y1 - y0]


def write_binary_mask(mask, path):
    from PIL import Image
    import numpy as np

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    mask_uint8 = (mask.astype("uint8") * 255)
    Image.fromarray(mask_uint8, mode="L").save(path)


def output_object_count(outputs):
    obj_ids = outputs.get("out_obj_ids", [])
    return len(obj_ids) if obj_ids is not None else 0


def merge_frame_outputs(frame_outputs, frame_idx, outputs):
    existing = frame_outputs.get(frame_idx)
    if existing is None or output_object_count(outputs) >= output_object_count(existing):
        frame_outputs[frame_idx] = outputs


def render_video_reference_frame(
    frame_image,
    frame_idx,
    frame_path,
    outputs,
    masks_dir,
    masked_frames_dir,
    bundle_root,
    prompt_text,
    used_explicit_geometry,
):
    from PIL import Image
    import numpy as np

    obj_ids = outputs.get("out_obj_ids", [])
    probs = outputs.get("out_probs", [])
    boxes_xywh = outputs.get("out_boxes_xywh", [])
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        binary_masks = []

    objects = []
    for obj_id, score, box_xywh, mask in zip(obj_ids, probs, boxes_xywh, binary_masks):
        mask_array = np.asarray(mask)
        if not mask_array.any():
            continue
        obj_id = int(obj_id)
        score = float(score)
        box_xywh = [float(value) for value in box_xywh]
        box_xyxy = [
            box_xywh[0],
            box_xywh[1],
            box_xywh[0] + box_xywh[2],
            box_xywh[1] + box_xywh[3],
        ]
        color = palette_color(obj_id)
        mask_path = masks_dir / f"frame_{frame_idx:06d}_obj_{obj_id:06d}.png"
        masked_frame_path = masked_frames_dir / f"frame_{frame_idx:06d}_obj_{obj_id:06d}.png"
        write_binary_mask(mask_array, mask_path)
        mask_image = Image.fromarray((mask_array.astype("uint8") * 255), mode="L")
        masked_frame = blend_mask_on_image(frame_image.convert("RGBA"), mask_image)
        draw_prediction_box(masked_frame, box_xyxy, color=color, score=score, index=obj_id)
        masked_frame.save(masked_frame_path)
        objects.append(
            {
                "obj_id": obj_id,
                "scores": [score],
                "presence_scores": None,
                "boxes_xyxy": [box_xyxy],
                "mask_path": str(mask_path.relative_to(bundle_root)),
                "masked_frame_path": str(masked_frame_path.relative_to(bundle_root)),
                "prompt_frame_idx": 0,
                "memory_frame_indices": [],
                "text_prompt": prompt_text,
                "used_explicit_geometry": used_explicit_geometry,
                "reused_previous_output": frame_idx != 0,
            }
        )

    return {
        "frame_idx": frame_idx,
        "frame_path": str(frame_path.relative_to(bundle_root)),
        "objects": objects,
    }


def main():
    args = parse_args()
    if (args.image is None) == (args.video is None):
        raise ValueError("provide exactly one of --image or --video")
    if args.box_label and len(args.box_label) != len(args.box):
        raise ValueError(
            f"--box-label count ({len(args.box_label)}) must match --box count ({len(args.box)})"
        )
    if args.video is not None and args.interactive_script is not None:
        raise ValueError("`--video` cannot be combined with `--interactive-script`")
    if args.video is not None and args.vision_only:
        raise ValueError("`--video` cannot be combined with `--vision-only`")
    if args.video is not None and args.debug_block:
        raise ValueError("`--video` cannot be combined with `--debug-block`")
    if args.video is not None:
        if args.prompt is None and not args.box:
            raise ValueError("video export requires --prompt, --box, or both")
        effective_prompt = args.prompt
    elif args.interactive_script is None:
        if args.prompt is None and not args.box:
            raise ValueError("provide --prompt, --box, or both")
        effective_prompt = args.prompt
        if effective_prompt is None and args.box:
            effective_prompt = "visual"
    else:
        if args.prompt is not None or args.box or args.box_label:
            raise ValueError(
                "`--interactive-script` currently derives the prompt flow internally; do not combine it with `--prompt`, `--box`, or `--box-label`"
            )
        if args.vision_only:
            raise ValueError("`--interactive-script` cannot be combined with `--vision-only`")
        if args.debug_block:
            raise ValueError("`--interactive-script` cannot be combined with `--debug-block`")
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
    script_path = (
        Path(args.interactive_script).expanduser().resolve()
        if args.interactive_script is not None
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    if args.video is not None:
        if device.type != "cuda":
            raise ValueError("upstream SAM3 video reference export currently requires CUDA")

        from PIL import Image

        from sam3.model_builder import build_sam3_predictor

        source_video_path = Path(args.video).expanduser().resolve()
        frames_dir = output_dir / "frames"
        masks_dir = output_dir / "masks"
        masked_frames_dir = output_dir / "masked_frames"
        masks_dir.mkdir(parents=True, exist_ok=True)
        masked_frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = prepare_video_frames(
            source_video_path, frames_dir, max_frames=args.video_frame_count
        )

        predictor = build_sam3_predictor(
            version="sam3",
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            compile=False,
            async_loading_frames=False,
            apply_temporal_disambiguation=args.video_apply_temporal_disambiguation,
        )
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir)}
        )
        session_id = response["session_id"]
        frame_outputs = {}

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
        }
        if args.prompt is not None:
            request["text"] = args.prompt
        if args.box:
            resolved_box_labels = [
                int(label)
                for label in (args.box_label if args.box_label else [True for _ in args.box])
            ]
            request["bounding_boxes"] = [box_cxcywh_to_xywh(box) for box in args.box]
            request["bounding_box_labels"] = resolved_box_labels
        prompt_response = predictor.handle_request(request)
        merge_frame_outputs(
            frame_outputs,
            int(prompt_response["frame_index"]),
            prompt_response["outputs"],
        )

        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": 0,
                "max_frame_num_to_track": len(frame_paths),
            }
        ):
            merge_frame_outputs(
                frame_outputs, int(response["frame_index"]), response["outputs"]
            )

        results = []
        for frame_idx, frame_path in enumerate(frame_paths):
            frame_image = Image.open(frame_path).convert("RGB")
            outputs = frame_outputs.get(
                frame_idx,
                {
                    "out_obj_ids": [],
                    "out_probs": [],
                    "out_boxes_xywh": [],
                    "out_binary_masks": [],
                },
            )
            results.append(
                render_video_reference_frame(
                    frame_image=frame_image,
                    frame_idx=frame_idx,
                    frame_path=frame_path,
                    outputs=outputs,
                    masks_dir=masks_dir,
                    masked_frames_dir=masked_frames_dir,
                    bundle_root=output_dir,
                    prompt_text=args.prompt,
                    used_explicit_geometry=bool(args.box),
                )
            )

        metadata = {
            "bundle_version": 1,
            "mode": "video_reference",
            "source_path": str(source_video_path),
            "source_kind": "video_file" if source_video_path.is_file() else "image_folder",
            "session_frame_count": len(frame_paths),
            "exported_frame_count": len(results),
            "frame_stride": 1,
            "tokenizer_path": (
                str(resolve_tokenizer_path(checkpoint_path))
                if resolve_tokenizer_path(checkpoint_path) is not None
                else None
            ),
            "prompt_text": args.prompt,
            "points_xy_normalized": [],
            "point_labels": [],
            "boxes_cxcywh_normalized": [list(box) for box in args.box],
            "box_labels": resolved_box_labels if args.box else [],
            "frames_dir": "frames",
            "masks_dir": "masks",
            "masked_frames_dir": "masked_frames",
            "results_path": "video_results.json",
            "video_apply_temporal_disambiguation": args.video_apply_temporal_disambiguation,
            "checkpoint_path": str(checkpoint_path),
            "bpe_path": str(bpe_path),
        }
        with open(output_dir / "video_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"saved video reference bundle to {output_dir}")
        print(f"  metadata: {output_dir / 'reference.json'}")
        print(f"  results: {output_dir / 'video_results.json'}")
        print(f"  frames: {frames_dir}")
        print(f"  masks: {masks_dir}")
        print(f"  masked frames: {masked_frames_dir}")
        return

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

    if script_path is not None:
        replay_steps = load_interactive_script(script_path)
        rendered_steps = []
        with torch.inference_mode():
            image_tensor = v2.functional.to_image(image).to(device)
            preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)
            backbone_out = model.backbone.forward_image(preprocessed_image)
            text_outputs = model.backbone.forward_text([effective_prompt], device=device)
            backbone_out.update(text_outputs)
            base_backbone_out = backbone_out
            find_input = processor.find_stage
            geometric_prompt = model._get_dummy_prompt()

            tensors = {"inputs.image": to_cpu_contiguous(preprocessed_image)}
            for step_idx, step in enumerate(replay_steps):
                print(
                    f"[interactive-export] step {step_idx + 1}/{len(replay_steps)} "
                    f"({step.get('name') or f'step_{step_idx:02}'})",
                    flush=True,
                )
                for point, label in zip(
                    step["step_points_xy_normalized"], step["step_point_labels"]
                ):
                    points = torch.tensor(point, device=device, dtype=torch.float32).view(1, 1, 2)
                    labels = torch.tensor([label], device=device, dtype=torch.long).view(1, 1)
                    geometric_prompt.append_points(points, labels)

                step_backbone_out = dict(base_backbone_out)
                prompt, prompt_mask, step_backbone_out = model._encode_prompt(
                    backbone_out=step_backbone_out,
                    find_input=find_input,
                    geometric_prompt=geometric_prompt.clone(),
                    encode_text=False,
                )
                step_backbone_out, encoder_out, _ = model._run_encoder(
                    backbone_out=step_backbone_out,
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
                    backbone_out=step_backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )

                tensors[f"step.{step_idx}.geometry.features"] = to_cpu_contiguous(prompt)
                tensors[f"step.{step_idx}.geometry.padding_mask"] = to_cpu_contiguous(
                    prompt_mask.to(torch.uint8)
                )
                tensors[f"step.{step_idx}.fusion.memory"] = to_cpu_contiguous(
                    encoder_out["encoder_hidden_states"]
                )
                tensors[f"step.{step_idx}.decoder.pred_logits"] = to_cpu_contiguous(
                    out["pred_logits"]
                )
                tensors[f"step.{step_idx}.decoder.pred_boxes_xyxy"] = to_cpu_contiguous(
                    out["pred_boxes_xyxy"]
                )
                tensors[f"step.{step_idx}.segmentation.mask_logits"] = to_cpu_contiguous(
                    out["pred_masks"]
                )
                if "presence_logit_dec" in out:
                    tensors[f"step.{step_idx}.decoder.presence_logits"] = to_cpu_contiguous(
                        out["presence_logit_dec"]
                    )
                rendered_steps.append(
                    render_interactive_reference_step(
                        image,
                        output_dir,
                        step_idx,
                        step.get("name") or f"step_{step_idx:02}",
                        step_idx,
                        args.image_size,
                        out["pred_logits"],
                        out["pred_boxes_xyxy"],
                        out["pred_masks"],
                        step["accumulated_points_xy_normalized"],
                        step["accumulated_point_labels"],
                    )
                )

        save_file(
            tensors,
            str(output_dir / "reference.safetensors"),
            metadata={"bundle_version": "1"},
        )
        metadata = {
            "bundle_version": 1,
            "mode": "interactive_reference",
            "image_path": str(Path(args.image).expanduser().resolve()),
            "image_size": args.image_size,
            "preprocess_mode": "exact",
            "replay_script_path": str(script_path),
            "effective_prompt": effective_prompt,
            "steps": replay_steps,
            "rendered_steps": rendered_steps,
            "checkpoint_path": str(checkpoint_path),
            "bpe_path": str(bpe_path),
        }
        with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"saved interactive reference bundle to {output_dir}")
        print(f"  tensors: {output_dir / 'reference.safetensors'}")
        print(f"  metadata: {output_dir / 'reference.json'}")
        print(f"  rendered steps: {len(rendered_steps)}")
        return

    with torch.inference_mode():
        image_tensor = v2.functional.to_image(image).to(device)
        preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)
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
                for model_box, label in zip(args.box, box_labels):
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
            torch.tensor(args.box, dtype=torch.float32)
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

        restored_mask, _raw_mask_probs = upsample_mask_to_original(
            out["pred_masks"][0, best_idx],
            args.image_size,
            (image.height, image.width),
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
            kept_mask, _ = upsample_mask_to_original(
                out["pred_masks"][0, kept_idx],
                args.image_size,
                (image.height, image.width),
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
        "box_labels": args.box_label if args.box_label else [True for _ in args.box],
        "image_size": args.image_size,
        "preprocess_mode": "exact",
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
