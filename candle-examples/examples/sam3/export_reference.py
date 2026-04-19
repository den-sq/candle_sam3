#!/usr/bin/env python3

import argparse
import contextlib
import inspect
import json
import shutil
import subprocess
import sys
import types
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
    parser.add_argument(
        "--video-scenario",
        default=None,
        help="Optional JSON scenario manifest describing the upstream video export engine, runtime overrides, and action sequence. When provided, it drives video export instead of the legacy one-prompt default flow.",
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
        "--video-debug-bundle",
        action="store_true",
        help="Write a focused video tracker debug bundle under <output-dir>/debug for prompt-frame and first-propagated-frame comparison.",
    )
    parser.add_argument(
        "--video-debug-obj-id",
        action="append",
        type=int,
        default=[],
        help="Restrict video debug export to the specified object id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--video-debug-frame",
        action="append",
        type=int,
        default=[],
        help="Restrict video debug export to the specified frame index. Can be passed multiple times.",
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


def load_video_scenario(path: Path):
    scenario = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError(f"video scenario {path} must be a JSON object")
    actions = scenario.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"video scenario {path} must contain a non-empty `actions` list")
    engine = scenario.get("engine", "video_inference")
    if engine not in {"video_inference", "tracker"}:
        raise ValueError(
            f"video scenario {path} has unsupported engine {engine!r}; expected `video_inference` or `tracker`"
        )
    return {
        "engine": engine,
        "apply_temporal_disambiguation": bool(
            scenario.get("apply_temporal_disambiguation", False)
        ),
        "tracker_overrides": dict(scenario.get("tracker_overrides", {})),
        "predictor_overrides": dict(scenario.get("predictor_overrides", {})),
        "session_overrides": dict(scenario.get("session_overrides", {})),
        "debug_capture_frame_indices": [
            int(frame_idx)
            for frame_idx in scenario.get("debug_capture_frame_indices", [])
        ],
        "debug_capture_obj_ids": [
            int(obj_id) for obj_id in scenario.get("debug_capture_obj_ids", [])
        ],
        "actions": actions,
        "raw": scenario,
    }


def build_default_video_scenario(args, resolved_box_labels):
    action = {
        "type": "add_prompt",
        "frame_idx": 0,
        "text": args.prompt,
    }
    if args.box:
        action["boxes_xywh"] = [box_cxcywh_to_xywh(box) for box in args.box]
        action["box_labels"] = [int(label) for label in resolved_box_labels]
    return {
        "engine": "video_inference",
        "apply_temporal_disambiguation": bool(args.video_apply_temporal_disambiguation),
        "tracker_overrides": {},
        "predictor_overrides": {},
        "session_overrides": {},
        "debug_capture_frame_indices": [],
        "debug_capture_obj_ids": [],
        "actions": [
            action,
            {
                "type": "propagate",
                "direction": "forward",
                "start_frame_idx": 0,
                "max_frame_num_to_track": int(args.video_frame_count),
            },
        ],
        "raw": None,
    }


def ensure_mapping(value, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def apply_attr_overrides(target, overrides, label):
    for key, value in ensure_mapping(overrides, label).items():
        if "." in key:
            current = target
            segments = key.split(".")
            for segment in segments[:-1]:
                if not hasattr(current, segment):
                    raise ValueError(
                        f"{label} override {key!r} refers to missing attribute {segment!r}"
                    )
                current = getattr(current, segment)
            leaf = segments[-1]
            if not hasattr(current, leaf):
                raise ValueError(
                    f"{label} override {key!r} refers to missing attribute {leaf!r}"
                )
            setattr(current, leaf, value)
        else:
            if not hasattr(target, key):
                raise ValueError(f"{label} override {key!r} refers to a missing attribute")
            setattr(target, key, value)


def capture_tracker_runtime_config(tracker):
    config = {
        "with_backbone": getattr(tracker, "backbone", None) is not None,
        "image_size": int(tracker.image_size),
        "backbone_stride": int(tracker.backbone_stride),
        "low_res_mask_size": int(tracker.low_res_mask_size),
        "input_mask_size": int(tracker.input_mask_size),
        "num_maskmem": int(tracker.num_maskmem),
        "max_cond_frames_in_attn": int(tracker.max_cond_frames_in_attn),
        "keep_first_cond_frame": bool(getattr(tracker, "keep_first_cond_frame", False)),
        "memory_temporal_stride_for_eval": int(
            getattr(tracker, "memory_temporal_stride_for_eval", 1)
        ),
        "max_obj_ptrs_in_encoder": int(tracker.max_obj_ptrs_in_encoder),
        "sigmoid_scale_for_mem_enc": float(tracker.sigmoid_scale_for_mem_enc),
        "sigmoid_bias_for_mem_enc": float(tracker.sigmoid_bias_for_mem_enc),
        "multimask_output_in_sam": bool(tracker.multimask_output_in_sam),
        "multimask_output_for_tracking": bool(tracker.multimask_output_for_tracking),
        "multimask_min_pt_num": int(tracker.multimask_min_pt_num),
        "multimask_max_pt_num": int(tracker.multimask_max_pt_num),
        "use_memory_selection": bool(tracker.use_memory_selection),
        "mf_threshold": float(tracker.mf_threshold),
        "forward_backbone_per_frame_for_eval": bool(
            getattr(tracker, "forward_backbone_per_frame_for_eval", False)
        ),
        "offload_output_to_cpu_for_eval": bool(
            getattr(tracker, "offload_output_to_cpu_for_eval", False)
        ),
        "trim_past_non_cond_mem_for_eval": bool(
            getattr(tracker, "trim_past_non_cond_mem_for_eval", False)
        ),
        "non_overlap_masks_for_mem_enc": bool(
            getattr(tracker, "non_overlap_masks_for_mem_enc", False)
        ),
        "sam_mask_decoder_extra_args": {
            "dynamic_multimask_via_stability": bool(
                getattr(tracker.sam_mask_decoder, "dynamic_multimask_via_stability", False)
            ),
            "dynamic_multimask_stability_delta": float(
                getattr(tracker.sam_mask_decoder, "dynamic_multimask_stability_delta", 0.0)
            ),
            "dynamic_multimask_stability_thresh": float(
                getattr(tracker.sam_mask_decoder, "dynamic_multimask_stability_thresh", 0.0)
            ),
        },
        "input_mask_binarize_threshold": 0.0,
        "video_mask_binarize_threshold": 0.5,
        "mask_as_output_out_scale": 20.0,
        "mask_as_output_out_bias": -10.0,
        "memory_prompt_mask_threshold": 0.0,
    }
    return config


def capture_predictor_runtime_config(model, tracker):
    return {
        "compile_model": bool(getattr(model, "compile_model", False)),
        "fill_hole_area": int(getattr(model, "fill_hole_area", 0)),
        "hotstart_delay": int(getattr(model, "hotstart_delay", 0)),
        "hotstart_unmatch_thresh": int(getattr(model, "hotstart_unmatch_thresh", 0)),
        "hotstart_dup_thresh": int(getattr(model, "hotstart_dup_thresh", 0)),
        "masklet_confirmation_enable": bool(
            getattr(model, "masklet_confirmation_enable", False)
        ),
        "masklet_confirmation_consecutive_det_thresh": int(
            getattr(model, "masklet_confirmation_consecutive_det_thresh", 0)
        ),
        "use_prev_mem_frame": bool(getattr(model, "use_prev_mem_frame", False)),
        "use_stateless_refinement": bool(
            getattr(model, "use_stateless_refinement", False)
        ),
        "refinement_detector_cond_frame_removal_window": int(
            getattr(model, "refinement_detector_cond_frame_removal_window", 0)
        ),
        "clear_non_cond_mem_around_input": bool(
            getattr(tracker, "clear_non_cond_mem_around_input", False)
        ),
        "clear_non_cond_mem_for_multi_obj": bool(
            getattr(tracker, "clear_non_cond_mem_for_multi_obj", False)
        ),
        "always_start_from_first_ann_frame": bool(
            getattr(tracker, "always_start_from_first_ann_frame", False)
        ),
        "max_point_num_in_prompt_enc": int(
            getattr(tracker, "max_point_num_in_prompt_enc", 0)
        ),
        "non_overlap_masks_for_output": bool(
            getattr(tracker, "non_overlap_masks_for_output", False)
        ),
        "iter_use_prev_mask_pred": bool(
            getattr(tracker, "iter_use_prev_mask_pred", False)
        ),
        "add_all_frames_to_correct_as_cond": bool(
            getattr(tracker, "add_all_frames_to_correct_as_cond", False)
        ),
    }


def build_upstream_video_predictor_for_export(
    build_sam3_predictor,
    checkpoint_path,
    bpe_path,
    apply_temporal_disambiguation,
    tracker_overrides,
    predictor_overrides,
):
    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        compile=False,
        async_loading_frames=False,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
    )
    model = predictor.model
    tracker = model.tracker
    apply_attr_overrides(tracker, tracker_overrides, "tracker_overrides")
    apply_attr_overrides(model, predictor_overrides, "predictor_overrides")
    return predictor


def scenario_uses_explicit_geometry(scenario):
    for action in scenario.get("actions", []):
        if any(
            action.get(key) is not None
            for key in ("boxes_xywh", "box_xyxy", "points_xy_normalized", "mask")
        ):
            return True
    return False


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


def prepare_tracker_input_frames(frame_paths, tracker_frames_dir: Path):
    from PIL import Image

    if tracker_frames_dir.exists():
        shutil.rmtree(tracker_frames_dir)
    tracker_frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, source_path in enumerate(frame_paths):
        frame = Image.open(source_path).convert("RGB")
        frame.save(
            tracker_frames_dir / f"{frame_idx:06d}.jpg",
            format="JPEG",
            quality=95,
        )
    tracker_frame_paths = sorted_frame_paths(tracker_frames_dir)
    if not tracker_frame_paths:
        raise RuntimeError(
            f"failed to prepare tracker input frames in {tracker_frames_dir}"
        )
    return tracker_frame_paths


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def box_cxcywh_to_xywh(box):
    x0, y0, x1, y1 = box_cxcywh_to_xyxy(box)
    return [x0, y0, x1 - x0, y1 - y0]


def binary_mask_to_box_xywh(mask):
    import numpy as np

    mask = np.asarray(mask).astype(bool)
    if mask.ndim == 3:
        mask = mask[0]
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    height, width = mask.shape
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    x0 = float(xs.min() / denom_x)
    y0 = float(ys.min() / denom_y)
    x1 = float(xs.max() / denom_x)
    y1 = float(ys.max() / denom_y)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def normalized_box_xyxy_to_mask(box_xyxy, width, height):
    import numpy as np

    x0, y0, x1, y1 = normalized_box_to_pixels(box_xyxy, width, height)
    mask = np.zeros((height, width), dtype="uint8")
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


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
        score = None if score is None else float(score)
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
                "scores": ([] if score is None else [score]),
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


def should_capture_video_debug(args, frame_idx, prompt_frame_idx=None):
    if args.video_debug_frame:
        return frame_idx in args.video_debug_frame
    if frame_idx == 0:
        return True
    if prompt_frame_idx is None:
        return False
    return frame_idx == prompt_frame_idx + 1


def should_capture_video_debug_obj(args, obj_id):
    return not args.video_debug_obj_id or int(obj_id) in args.video_debug_obj_id


def build_video_debug_prompt_metadata_from_action(action, default_text_prompt=None):
    boxes_xywh = action.get("boxes_xywh", [])
    box_labels = action.get("box_labels", [1 for _ in boxes_xywh])
    points = action.get("points_xy_normalized", [])
    point_labels = action.get("point_labels", [])
    return {
        "text_prompt": action.get("text", default_text_prompt),
        "used_visual_text_prompt": action.get("text", default_text_prompt) is None
        and len(boxes_xywh) == 1,
        "normalized_points_xy": [list(point) for point in points],
        "point_labels": [int(label) for label in point_labels],
        "normalized_boxes_xywh": [list(box) for box in boxes_xywh],
        "normalized_box_xyxy": (
            [list(action["box_xyxy"])]
            if action.get("box_xyxy") is not None
            else []
        ),
        "box_labels": [int(label) for label in box_labels],
        "mask_prompt": action.get("mask"),
    }


def write_video_debug_binary_mask(mask, path):
    write_binary_mask(mask, path)


def build_video_debug_observable(
    output_dir,
    frame_idx,
    obj_id,
    stage_suffix,
    mask,
    score,
    box_xywh,
):
    import numpy as np

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[0]
    mask_array = mask_array.astype("uint8")
    foreground_pixel_count = int(mask_array.sum())
    mask_path = output_dir / f"frame_{frame_idx:06d}_obj_{int(obj_id):06d}_{stage_suffix}.png"
    write_video_debug_binary_mask(mask_array, mask_path)
    total_pixels = max(mask_array.size, 1)
    return {
        "mask_path": str(mask_path.relative_to(output_dir)),
        "mask_threshold": 0.5,
        "foreground_pixel_count": foreground_pixel_count,
        "mask_area_ratio": foreground_pixel_count / total_pixels,
        "boxes_xyxy": [[
            float(box_xywh[0]),
            float(box_xywh[1]),
            float(box_xywh[0] + box_xywh[2]),
            float(box_xywh[1] + box_xywh[3]),
        ]],
        "scores": ([] if score is None else [float(score)]),
        "presence_scores": None,
        "mask_logits_stats": {
            "shape": list(mask_array.shape),
            "dtype": str(mask_array.dtype),
            "min": float(mask_array.min()) if mask_array.size else 0.0,
            "max": float(mask_array.max()) if mask_array.size else 0.0,
            "mean": float(mask_array.mean()) if mask_array.size else 0.0,
            "l2_norm": float(np.linalg.norm(mask_array.astype("float32"))),
            "foreground_pixel_count": foreground_pixel_count,
        },
        "mask_prob_stats": {
            "shape": list(mask_array.shape),
            "dtype": str(mask_array.dtype),
            "min": float(mask_array.min()) if mask_array.size else 0.0,
            "max": float(mask_array.max()) if mask_array.size else 0.0,
            "mean": float(mask_array.mean()) if mask_array.size else 0.0,
            "l2_norm": float(np.linalg.norm(mask_array.astype("float32"))),
            "foreground_pixel_count": foreground_pixel_count,
        },
    }


def build_tensor_stats(tensor):
    tensor = tensor.detach().to("cpu").contiguous()
    stats_tensor = tensor.float() if not tensor.dtype.is_floating_point else tensor
    numel = tensor.numel()
    if numel == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "l2_norm": 0.0,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
        "mean": float(stats_tensor.mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(stats_tensor).item()),
    }


def clone_tensor_for_fixture(tensor):
    return tensor.detach().to("cpu").contiguous()


def add_tensor_tree(target, prefix, value):
    import numpy as np

    if value is None:
        return
    if torch.is_tensor(value):
        target[prefix] = value
        return
    if isinstance(value, np.ndarray):
        target[prefix] = torch.from_numpy(np.ascontiguousarray(value))
        return
    if hasattr(value, "_asdict"):
        add_tensor_tree(target, prefix, value._asdict())
        return
    if isinstance(value, dict):
        for key, item in value.items():
            add_tensor_tree(target, f"{prefix}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            add_tensor_tree(target, f"{prefix}.{idx}", item)


def add_tracker_output_tensors(target, prefix, frame_output):
    if frame_output is None:
        return
    add_tensor_tree(target, prefix, frame_output)


class VideoInternalFixtureRecorder:
    def __init__(self, output_dir: Path, capture_frame_indices):
        self.output_dir = output_dir
        self.capture_frame_indices = sorted({int(frame_idx) for frame_idx in capture_frame_indices})
        self.tensor_store = {}
        self.records = []
        self.session_id = None
        self.tracker_config = None
        self.predictor_config = None
        self.engine = None
        self.scenario = None
        self._context_stack = []

    def should_capture(self, frame_idx):
        return frame_idx in self.capture_frame_indices

    def set_session_id(self, session_id):
        self.session_id = session_id

    def set_tracker_config(self, tracker):
        self.tracker_config = capture_tracker_runtime_config(tracker)

    def set_predictor_config(self, model, tracker):
        self.predictor_config = capture_predictor_runtime_config(model, tracker)

    def set_engine(self, engine):
        self.engine = engine

    def set_scenario(self, scenario):
        self.scenario = scenario

    def push_context(self, **context):
        self._context_stack.append(context)

    def pop_context(self):
        if self._context_stack:
            self._context_stack.pop()

    def current_context(self):
        return self._context_stack[-1] if self._context_stack else {}

    def add_record(self, stage, frame_idx, metadata=None, tensors=None):
        frame_idx = int(frame_idx)
        if not self.should_capture(frame_idx):
            return

        record_idx = len(self.records)
        tensor_keys = {}
        tensor_stats = {}
        for name, tensor in (tensors or {}).items():
            if tensor is None or not torch.is_tensor(tensor):
                continue
            tensor_key = f"record_{record_idx:04d}.{name}"
            stored = clone_tensor_for_fixture(tensor)
            self.tensor_store[tensor_key] = stored
            tensor_keys[name] = tensor_key
            tensor_stats[name] = build_tensor_stats(stored)

        self.records.append(
            {
                "record_index": record_idx,
                "stage": stage,
                "frame_idx": frame_idx,
                "metadata": metadata or {},
                "tensor_keys": tensor_keys,
                "tensor_stats": tensor_stats,
            }
        )

    def finalize(self):
        from safetensors.torch import save_file

        fixtures_path = self.output_dir / "internal_fixtures.safetensors"
        manifest_path = self.output_dir / "internal_manifest.json"
        if self.tensor_store:
            save_file(self.tensor_store, fixtures_path)

        manifest = {
            "bundle_version": 1,
            "mode": "video_internal_fixtures",
            "source": "upstream",
            "engine": self.engine,
            "session_id": self.session_id,
            "capture_frame_indices": list(self.capture_frame_indices),
            "tensor_file": fixtures_path.name if self.tensor_store else None,
            "tracker_config": self.tracker_config,
            "predictor_config": self.predictor_config,
            "scenario": self.scenario,
            "records": self.records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path, fixtures_path if self.tensor_store else None


def compute_memory_selection_debug(tracker, frame_idx, output_dict, num_frames, track_in_reverse, use_prev_mem_frame):
    from sam3.model.sam3_tracker_utils import select_closest_cond_frames

    selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
        frame_idx,
        output_dict["cond_frame_outputs"],
        tracker.max_cond_frames_in_attn,
        keep_first_cond_frame=tracker.keep_first_cond_frame,
    )
    metadata = {
        "selected_conditioning_frame_indices": sorted(int(idx) for idx in selected_cond_outputs.keys()),
        "unselected_conditioning_frame_indices": sorted(int(idx) for idx in unselected_cond_outputs.keys()),
        "selected_memory_frame_indices": [],
        "selected_memory_sources": [],
        "selected_object_pointer_frame_indices": [],
        "selected_object_pointer_is_conditioning": [],
        "selected_object_pointer_temporal_offsets": [],
        "memory_temporal_stride": int(1 if tracker.training else tracker.memory_temporal_stride_for_eval),
        "use_prev_mem_frame": bool(use_prev_mem_frame),
        "track_in_reverse": bool(track_in_reverse),
        "max_obj_ptrs_in_encoder": int(min(num_frames, tracker.max_obj_ptrs_in_encoder)),
    }
    tensor_map = {}
    if not use_prev_mem_frame:
        return metadata, tensor_map

    r = metadata["memory_temporal_stride"]
    valid_indices = None
    if tracker.use_memory_selection:
        valid_indices = tracker.frame_filter(output_dict, track_in_reverse, frame_idx, num_frames, r)
        metadata["memory_selection_valid_indices"] = [int(idx) for idx in valid_indices]

    for t_pos in range(1, tracker.num_maskmem):
        t_rel = tracker.num_maskmem - t_pos
        if tracker.use_memory_selection:
            if t_rel > len(valid_indices):
                continue
            prev_frame_idx = int(valid_indices[-t_rel])
        else:
            if t_rel == 1:
                prev_frame_idx = int(frame_idx + t_rel if track_in_reverse else frame_idx - t_rel)
            else:
                if not track_in_reverse:
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                prev_frame_idx = int(prev_frame_idx)
        source = "non_cond"
        out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
        if out is None:
            out = unselected_cond_outputs.get(prev_frame_idx)
            if out is not None:
                source = "unselected_cond"
        if out is None:
            continue
        metadata["selected_memory_frame_indices"].append(prev_frame_idx)
        metadata["selected_memory_sources"].append(source)
        add_tracker_output_tensors(tensor_map, f"selected_memory_frames.{prev_frame_idx}", out)

    if not tracker.training:
        ptr_cond_outputs = {
            t: out
            for t, out in selected_cond_outputs.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        }
    else:
        ptr_cond_outputs = selected_cond_outputs
    pos_and_ptrs = [
        (
            int((frame_idx - t) * (-1 if track_in_reverse else 1)),
            int(t),
            out["obj_ptr"],
            True,
        )
        for t, out in ptr_cond_outputs.items()
    ]
    max_obj_ptrs_in_encoder = min(num_frames, tracker.max_obj_ptrs_in_encoder)
    for t_diff in range(1, max_obj_ptrs_in_encoder):
        if not tracker.use_memory_selection:
            frame_t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if frame_t < 0 or (num_frames is not None and frame_t >= num_frames):
                break
        else:
            if -t_diff <= -len(valid_indices):
                break
            frame_t = int(valid_indices[-t_diff])
        out = output_dict["non_cond_frame_outputs"].get(frame_t, unselected_cond_outputs.get(frame_t))
        if out is not None:
            pos_and_ptrs.append((int(t_diff), int(frame_t), out["obj_ptr"], False))

    if pos_and_ptrs:
        pos_list = [item[0] for item in pos_and_ptrs]
        metadata["selected_object_pointer_frame_indices"] = [item[1] for item in pos_and_ptrs]
        metadata["selected_object_pointer_is_conditioning"] = [bool(item[3]) for item in pos_and_ptrs]
        metadata["selected_object_pointer_temporal_offsets"] = list(pos_list)
        tensor_map["object_pointer_temporal_pos_enc"] = tracker._get_tpos_enc(
            pos_list,
            max_abs_pos=max_obj_ptrs_in_encoder,
            device=pos_and_ptrs[0][2].device,
        )
        for _, ptr_frame_idx, obj_ptr, _ in pos_and_ptrs:
            tensor_map[f"selected_object_pointer_frames.{ptr_frame_idx}.obj_ptr"] = obj_ptr

    for cond_frame_idx, cond_output in selected_cond_outputs.items():
        add_tracker_output_tensors(
            tensor_map,
            f"selected_conditioning_frames.{int(cond_frame_idx)}",
            cond_output,
        )

    return metadata, tensor_map


def install_video_internal_fixture_recorder(target, debug_dir: Path, capture_frame_indices, scenario=None):
    recorder = VideoInternalFixtureRecorder(debug_dir, capture_frame_indices)
    if hasattr(target, "model") and hasattr(target.model, "tracker"):
        predictor = target
        model = target.model
        predictor_impl = model
        tracker = model.tracker
        engine = "video_inference"
    else:
        predictor = target
        model = None
        predictor_impl = target
        tracker = target
        engine = "tracker"
    recorder.set_engine(engine)
    if scenario is not None:
        recorder.set_scenario(scenario)
    recorder.set_tracker_config(tracker)
    if model is not None:
        recorder.set_predictor_config(model, tracker)

    original_get_visual_prompt = getattr(predictor_impl, "_get_visual_prompt", None)
    original_get_image_feature = getattr(predictor_impl, "_get_image_feature", None)
    original_predictor_run_single_frame_inference = getattr(
        predictor_impl, "_run_single_frame_inference", None
    )
    original_build_tracker_output = getattr(predictor_impl, "_build_tracker_output", None)
    original_postprocess_output = getattr(predictor_impl, "_postprocess_output", None)
    original_preflight = getattr(predictor_impl, "propagate_in_video_preflight", None)
    original_clear_non_cond_mem = getattr(
        predictor_impl, "_clear_non_cond_mem_around_input", None
    )
    original_tracker_add_new_objects = getattr(model, "_tracker_add_new_objects", None)
    original_det_track_one_frame = getattr(model, "_det_track_one_frame", None)
    original_run_memory_encoder = tracker._run_memory_encoder
    original_prepare_memory_conditioned_features = tracker._prepare_memory_conditioned_features
    original_track_step = tracker.track_step
    original_encode_new_memory = tracker._encode_new_memory
    original_forward_sam_heads = getattr(tracker, "_forward_sam_heads", None)
    original_use_mask_as_output = getattr(tracker, "_use_mask_as_output", None)
    original_tracker_forward_image = getattr(tracker, "forward_image", None)
    original_add_new_points_or_box = getattr(tracker, "add_new_points_or_box", None)
    original_add_new_mask = getattr(tracker, "add_new_mask", None)
    original_prompt_encoder_forward = getattr(
        getattr(tracker, "sam_prompt_encoder", None), "forward", None
    )
    original_mask_decoder_forward = getattr(
        getattr(tracker, "sam_mask_decoder", None), "forward", None
    )

    def bind_call_arguments(method, bound_self, args, kwargs):
        target = method.__func__ if hasattr(method, "__func__") else method
        signature = inspect.signature(target)
        return signature.bind_partial(bound_self, *args, **kwargs).arguments

    def sanitize_sequence(values):
        if values is None:
            return None
        return [int(value) for value in values]

    def add_result_tree(tensor_map, prefix, value):
        add_tensor_tree(tensor_map, prefix, value)

    def output_frame_keys(output_dict):
        if output_dict is None:
            return {"cond_frame_outputs": [], "non_cond_frame_outputs": []}
        return {
            "cond_frame_outputs": sorted(
                int(frame_idx) for frame_idx in output_dict.get("cond_frame_outputs", {}).keys()
            ),
            "non_cond_frame_outputs": sorted(
                int(frame_idx) for frame_idx in output_dict.get("non_cond_frame_outputs", {}).keys()
            ),
        }

    def object_level_output_frame_keys(output_dict_per_obj):
        frame_keys = {}
        if output_dict_per_obj is None:
            return frame_keys
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            frame_keys[str(int(obj_idx))] = output_frame_keys(obj_output_dict)
        return frame_keys

    def record_for_context(stage, metadata=None, tensors=None):
        frame_idx = recorder.current_context().get("frame_idx")
        if frame_idx is None or not recorder.should_capture(frame_idx):
            return
        recorder.add_record(
            stage=stage,
            frame_idx=frame_idx,
            metadata=metadata,
            tensors=tensors,
        )

    def wrapped_get_visual_prompt(self, *args, **kwargs):
        call_args = bind_call_arguments(original_get_visual_prompt, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        result = original_get_visual_prompt(*args, **kwargs)
        if recorder.should_capture(frame_idx):
            tensor_map = {}
            tensor_map["boxes_cxcywh_in"] = call_args["boxes_cxcywh"]
            tensor_map["box_labels_in"] = call_args["box_labels"]
            add_result_tree(
                tensor_map,
                "visual_prompt_result",
                {
                    "boxes_cxcywh_out": result[0],
                    "box_labels_out": result[1],
                    "new_visual_prompt": result[2],
                },
            )
            recorder.add_record(
                stage="get_visual_prompt",
                frame_idx=frame_idx,
                metadata={
                    "had_previous_outputs": call_args["inference_state"]["previous_stages_out"][frame_idx]
                    is not None,
                    "had_existing_visual_prompt": call_args["inference_state"]["per_frame_visual_prompt"][frame_idx]
                    is not None,
                    "input_box_count": int(call_args["boxes_cxcywh"].shape[0]),
                    "returned_box_count": int(result[0].shape[0]),
                    "created_visual_prompt": result[2] is not None,
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_predictor_run_single_frame_inference(self, *args, **kwargs):
        call_args = bind_call_arguments(
            original_predictor_run_single_frame_inference, self, args, kwargs
        )
        frame_idx = int(call_args["frame_idx"])
        call_args["inference_state"]["_strict_port_debug_last_frame_idx"] = frame_idx
        recorder.push_context(stage="run_single_frame_inference", frame_idx=frame_idx)
        try:
            result = original_predictor_run_single_frame_inference(*args, **kwargs)
        finally:
            recorder.pop_context()
        if recorder.should_capture(frame_idx):
            inference_state = call_args["inference_state"]
            tensor_map = {}
            add_result_tree(tensor_map, "run_single_frame_inference_output", result)
            recorder.add_record(
                stage="run_single_frame_inference",
                frame_idx=frame_idx,
                metadata={
                    "reverse": bool(call_args.get("reverse", False)),
                    "tracking_has_started": bool(
                        inference_state.get("tracking_has_started", False)
                    ),
                    "first_ann_frame_idx": (
                        None
                        if inference_state.get("first_ann_frame_idx") is None
                        else int(inference_state["first_ann_frame_idx"])
                    ),
                    "cached_frame_output_indices": sorted(
                        int(idx)
                        for idx in inference_state.get("cached_frame_outputs", {}).keys()
                    ),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_get_image_feature(self, *args, **kwargs):
        call_args = bind_call_arguments(original_get_image_feature, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        inference_state = call_args["inference_state"]
        cache_hit = inference_state["cached_features"].get(frame_idx, (None, None))[1] is not None
        recorder.push_context(
            stage="get_image_feature",
            frame_idx=frame_idx,
            batch_size=int(call_args["batch_size"]),
        )
        try:
            result = original_get_image_feature(*args, **kwargs)
        finally:
            recorder.pop_context()
        if recorder.should_capture(frame_idx):
            image, backbone_out, current_vision_feats, current_vision_pos_embeds, feat_sizes = result
            tensor_map = {"image": image}
            add_result_tree(tensor_map, "backbone_out", backbone_out)
            add_result_tree(tensor_map, "current_vision_feats", current_vision_feats)
            add_result_tree(
                tensor_map,
                "current_vision_pos_embeds",
                current_vision_pos_embeds,
            )
            recorder.add_record(
                stage="get_image_feature",
                frame_idx=frame_idx,
                metadata={
                    "batch_size": int(call_args["batch_size"]),
                    "cache_hit": bool(cache_hit),
                    "feat_sizes": [[int(h), int(w)] for h, w in feat_sizes],
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_build_tracker_output(self, *args, **kwargs):
        call_args = bind_call_arguments(original_build_tracker_output, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        result = original_build_tracker_output(*args, **kwargs)
        if recorder.should_capture(frame_idx):
            tensor_map = {}
            if call_args.get("refined_obj_id_to_mask") is not None:
                add_result_tree(
                    tensor_map,
                    "refined_obj_id_to_mask",
                    call_args["refined_obj_id_to_mask"],
                )
            add_result_tree(tensor_map, "build_tracker_output", result)
            recorder.add_record(
                stage="build_tracker_output",
                frame_idx=frame_idx,
                metadata={
                    "cached_frame_output_indices": sorted(
                        int(idx)
                        for idx in call_args["inference_state"].get(
                            "cached_frame_outputs", {}
                        ).keys()
                    ),
                    "has_refined_masks": call_args.get("refined_obj_id_to_mask") is not None,
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_postprocess_output(self, *args, **kwargs):
        call_args = bind_call_arguments(original_postprocess_output, self, args, kwargs)
        inference_state = call_args["inference_state"]
        out = call_args["out"]
        result = original_postprocess_output(*args, **kwargs)
        frame_stats = out.get("frame_stats", None)
        frame_idx = None
        if frame_stats is not None and isinstance(frame_stats, dict):
            raw_frame_idx = frame_stats.get("frame_idx")
            if raw_frame_idx is not None:
                frame_idx = int(raw_frame_idx)
        if frame_idx is None:
            frame_idx = inference_state.get("_strict_port_debug_last_frame_idx")
        if frame_idx is None:
            frame_idx = recorder.current_context().get("frame_idx")
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {}
            add_result_tree(tensor_map, "postprocess_input", out)
            add_result_tree(tensor_map, "postprocess_output", result)
            recorder.add_record(
                stage="postprocess_output",
                frame_idx=frame_idx,
                metadata={
                    "removed_obj_ids": sanitize_sequence(call_args.get("removed_obj_ids")),
                    "suppressed_obj_ids": sanitize_sequence(call_args.get("suppressed_obj_ids")),
                    "unconfirmed_obj_ids": sanitize_sequence(call_args.get("unconfirmed_obj_ids")),
                    "fill_hole_area": int(getattr(self, "fill_hole_area", 0)),
                    "non_overlap_masks_for_output": bool(
                        getattr(self, "non_overlap_masks_for_output", False)
                    ),
                    "orig_video_height": int(inference_state["orig_height"]),
                    "orig_video_width": int(inference_state["orig_width"]),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_preflight(self, *args, **kwargs):
        call_args = bind_call_arguments(original_preflight, self, args, kwargs)
        inference_state = call_args["inference_state"]
        before_output_keys = output_frame_keys(inference_state.get("output_dict"))
        before_per_obj_keys = object_level_output_frame_keys(
            inference_state.get("output_dict_per_obj")
        )
        before_tracking_has_started = bool(inference_state.get("tracking_has_started", False))
        result = original_preflight(*args, **kwargs)
        after_output_dict = inference_state.get("output_dict", {})
        after_output_keys = output_frame_keys(after_output_dict)
        consolidated = inference_state.get("consolidated_frame_inds", {})
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for frame_idx, frame_output in after_output_dict.get(storage_key, {}).items():
                if not recorder.should_capture(frame_idx):
                    continue
                tensor_map = {}
                add_tracker_output_tensors(
                    tensor_map,
                    f"preflight_output.{storage_key}.{int(frame_idx)}",
                    frame_output,
                )
                recorder.add_record(
                    stage="propagate_in_video_preflight",
                    frame_idx=int(frame_idx),
                    metadata={
                        "run_mem_encoder": bool(call_args.get("run_mem_encoder", True)),
                        "before_output_frame_keys": before_output_keys,
                        "after_output_frame_keys": after_output_keys,
                        "before_per_obj_output_frame_keys": before_per_obj_keys,
                        "after_per_obj_output_frame_keys": object_level_output_frame_keys(
                            inference_state.get("output_dict_per_obj")
                        ),
                        "consolidated_cond_frame_indices": sorted(
                            int(idx)
                            for idx in consolidated.get("cond_frame_outputs", set())
                        ),
                        "consolidated_non_cond_frame_indices": sorted(
                            int(idx)
                            for idx in consolidated.get("non_cond_frame_outputs", set())
                        ),
                        "tracking_has_started_before": before_tracking_has_started,
                        "tracking_has_started_after": bool(
                            inference_state.get("tracking_has_started", False)
                        ),
                        "first_ann_frame_idx": (
                            None
                            if inference_state.get("first_ann_frame_idx") is None
                            else int(inference_state["first_ann_frame_idx"])
                        ),
                    },
                    tensors=tensor_map,
                )
        return result

    def wrapped_clear_non_cond_mem(self, *args, **kwargs):
        call_args = bind_call_arguments(original_clear_non_cond_mem, self, args, kwargs)
        inference_state = call_args["inference_state"]
        frame_idx = int(call_args["frame_idx"])
        before_non_cond = object_level_output_frame_keys(
            inference_state.get("output_dict_per_obj")
        )
        result = original_clear_non_cond_mem(*args, **kwargs)
        if recorder.should_capture(frame_idx):
            recorder.add_record(
                stage="clear_non_cond_mem_around_input",
                frame_idx=frame_idx,
                metadata={
                    "memory_temporal_stride_for_eval": int(
                        getattr(self, "memory_temporal_stride_for_eval", 1)
                    ),
                    "num_maskmem": int(getattr(self, "num_maskmem", 0)),
                    "before_per_obj_output_frame_keys": before_non_cond,
                    "after_per_obj_output_frame_keys": object_level_output_frame_keys(
                        inference_state.get("output_dict_per_obj")
                    ),
                },
                tensors={},
            )
        return result

    def wrapped_det_track_one_frame(self, *args, **kwargs):
        call_args = bind_call_arguments(original_det_track_one_frame, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        recorder.push_context(stage="det_track_one_frame", frame_idx=frame_idx)
        try:
            result = original_det_track_one_frame(*args, **kwargs)
        finally:
            recorder.pop_context()
        if recorder.should_capture(frame_idx):
            tensor_map = {}
            add_result_tree(tensor_map, "det_track_one_frame_output", result)
            recorder.add_record(
                stage="det_track_one_frame",
                frame_idx=frame_idx,
                metadata={
                    "num_frames": int(call_args["num_frames"]),
                    "reverse": bool(call_args["reverse"]),
                    "allow_new_detections": bool(
                        call_args.get("allow_new_detections", True)
                    ),
                    "is_image_only": bool(call_args.get("is_image_only", False)),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_tracker_add_new_objects(self, *args, **kwargs):
        call_args = bind_call_arguments(original_tracker_add_new_objects, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        if recorder.should_capture(frame_idx):
            recorder.add_record(
                stage="tracker_add_new_objects_input",
                frame_idx=frame_idx,
                metadata={
                    "num_frames": int(call_args["num_frames"]),
                    "new_obj_ids": [int(obj_id) for obj_id in call_args["new_obj_ids"]],
                    "orig_video_height": int(call_args["orig_vid_height"]),
                    "orig_video_width": int(call_args["orig_vid_width"]),
                    "input_mask_size": int(self.tracker.input_mask_size),
                    "input_mask_binarize_threshold": 0.0,
                },
                tensors={"new_object_masks_before_resize": call_args["new_obj_masks"]},
            )
        result = original_tracker_add_new_objects(*args, **kwargs)
        if recorder.should_capture(frame_idx) and result:
            tracker_state = result[-1]
            cond_output = tracker_state["output_dict"]["cond_frame_outputs"].get(frame_idx)
            tensor_map = {}
            add_tracker_output_tensors(tensor_map, "post_preflight_cond_output", cond_output)
            recorder.add_record(
                stage="tracker_add_new_objects_post_preflight",
                frame_idx=frame_idx,
                metadata={
                    "num_tracker_states_local": len(result),
                    "tracked_obj_ids": [int(obj_id) for obj_id in tracker_state.get("obj_ids", [])],
                    "frames_already_tracked": sorted(int(idx) for idx in tracker_state.get("frames_already_tracked", {}).keys()),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_add_new_points_or_box(self, *args, **kwargs):
        call_args = bind_call_arguments(original_add_new_points_or_box, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        if recorder.should_capture(frame_idx):
            tensors = {}
            if call_args.get("points") is not None:
                tensors["points"] = call_args["points"]
            if call_args.get("labels") is not None:
                tensors["labels"] = call_args["labels"]
            if call_args.get("box") is not None:
                tensors["box"] = call_args["box"]
            recorder.add_record(
                stage="add_new_points_or_box_input",
                frame_idx=frame_idx,
                metadata={
                    "obj_id": int(call_args["obj_id"]),
                    "clear_old_points": bool(call_args.get("clear_old_points", True)),
                    "rel_coordinates": bool(call_args.get("rel_coordinates", True)),
                    "use_prev_mem_frame": bool(call_args.get("use_prev_mem_frame", False)),
                },
                tensors=tensors,
            )
        return original_add_new_points_or_box(*args, **kwargs)

    def wrapped_add_new_mask(self, *args, **kwargs):
        call_args = bind_call_arguments(original_add_new_mask, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        if recorder.should_capture(frame_idx):
            recorder.add_record(
                stage="add_new_mask_input",
                frame_idx=frame_idx,
                metadata={
                    "obj_id": int(call_args["obj_id"]),
                    "add_mask_to_memory": bool(call_args.get("add_mask_to_memory", False)),
                },
                tensors={"mask": call_args["mask"]},
            )
        return original_add_new_mask(*args, **kwargs)

    def wrapped_run_memory_encoder(self, *args, **kwargs):
        call_args = bind_call_arguments(original_run_memory_encoder, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        recorder.push_context(stage="run_memory_encoder", frame_idx=frame_idx)
        try:
            result = original_run_memory_encoder(*args, **kwargs)
        finally:
            recorder.pop_context()
        if recorder.should_capture(frame_idx):
            maskmem_features, maskmem_pos_enc = result
            tensor_map = {
                "high_res_masks": call_args["high_res_masks"],
                "object_score_logits": call_args["object_score_logits"],
                "maskmem_features": maskmem_features,
            }
            for idx, pos_enc in enumerate(maskmem_pos_enc):
                tensor_map[f"maskmem_pos_enc.{idx}"] = pos_enc
            recorder.add_record(
                stage="run_memory_encoder",
                frame_idx=frame_idx,
                metadata={
                    "batch_size": int(call_args["batch_size"]),
                    "is_mask_from_pts": bool(call_args["is_mask_from_pts"]),
                    "sigmoid_scale_for_mem_enc": float(self.sigmoid_scale_for_mem_enc),
                    "sigmoid_bias_for_mem_enc": float(self.sigmoid_bias_for_mem_enc),
                    "mask_for_mem_threshold": 0.0 if call_args["is_mask_from_pts"] and not self.training else None,
                    "mask_for_mem_mode": "binary_gt_0" if call_args["is_mask_from_pts"] and not self.training else "sigmoid_logits",
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_prepare_memory_conditioned_features(self, *args, **kwargs):
        call_args = bind_call_arguments(
            original_prepare_memory_conditioned_features, self, args, kwargs
        )
        frame_idx = int(call_args["frame_idx"])
        metadata = None
        tensor_map = None
        if recorder.should_capture(frame_idx):
            metadata, tensor_map = compute_memory_selection_debug(
                tracker=self,
                frame_idx=frame_idx,
                output_dict=call_args["output_dict"],
                num_frames=call_args["num_frames"],
                track_in_reverse=call_args.get("track_in_reverse", False),
                use_prev_mem_frame=call_args.get("use_prev_mem_frame", True),
            )
        result = original_prepare_memory_conditioned_features(*args, **kwargs)
        if recorder.should_capture(frame_idx):
            capture_tensors = dict(tensor_map or {})
            capture_tensors["pix_feat_with_mem"] = result
            recorder.add_record(
                stage="prepare_memory_conditioned_features",
                frame_idx=frame_idx,
                metadata=metadata,
                tensors=capture_tensors,
            )
        return result

    def wrapped_track_step(self, *args, **kwargs):
        call_args = bind_call_arguments(original_track_step, self, args, kwargs)
        frame_idx = int(call_args["frame_idx"])
        recorder.push_context(
            stage="track_step",
            frame_idx=frame_idx,
            is_init_cond_frame=bool(call_args["is_init_cond_frame"]),
            run_mem_encoder=bool(call_args.get("run_mem_encoder", True)),
        )
        try:
            result = original_track_step(*args, **kwargs)
        finally:
            recorder.pop_context()
        if recorder.should_capture(frame_idx):
            point_inputs = call_args.get("point_inputs")
            mask_inputs = call_args.get("mask_inputs")
            tensor_map = {
                "current_vision_feats": call_args["current_vision_feats"][-1],
                "current_vision_pos_embeds": call_args["current_vision_pos_embeds"][-1],
            }
            if mask_inputs is not None:
                tensor_map["mask_inputs"] = mask_inputs
            if point_inputs is not None:
                tensor_map["point_coords"] = point_inputs["point_coords"]
                tensor_map["point_labels"] = point_inputs["point_labels"]
            add_tracker_output_tensors(tensor_map, "track_step_output", result)
            recorder.add_record(
                stage="track_step",
                frame_idx=frame_idx,
                metadata={
                    "is_init_cond_frame": bool(call_args["is_init_cond_frame"]),
                    "track_in_reverse": bool(call_args.get("track_in_reverse", False)),
                    "run_mem_encoder": bool(call_args.get("run_mem_encoder", True)),
                    "use_prev_mem_frame": bool(call_args.get("use_prev_mem_frame", True)),
                    "num_frames": int(call_args["num_frames"]),
                    "point_input_count": int(point_inputs["point_labels"].numel()) if point_inputs is not None else 0,
                    "has_mask_inputs": mask_inputs is not None,
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_forward_sam_heads(self, *args, **kwargs):
        call_args = bind_call_arguments(original_forward_sam_heads, self, args, kwargs)
        frame_idx = recorder.current_context().get("frame_idx")
        if frame_idx is not None:
            recorder.push_context(stage="forward_sam_heads", frame_idx=frame_idx)
        try:
            result = original_forward_sam_heads(*args, **kwargs)
        finally:
            if frame_idx is not None:
                recorder.pop_context()
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {
                "backbone_features": call_args["backbone_features"],
                "image_pe": self.sam_prompt_encoder.get_dense_pe(),
            }
            if call_args.get("point_inputs") is not None:
                add_result_tree(tensor_map, "point_inputs", call_args["point_inputs"])
            if call_args.get("mask_inputs") is not None:
                tensor_map["mask_inputs"] = call_args["mask_inputs"]
            if call_args.get("high_res_features") is not None:
                add_result_tree(
                    tensor_map, "high_res_features", call_args["high_res_features"]
                )
            add_result_tree(
                tensor_map,
                "forward_sam_heads_output",
                {
                    "low_res_multimasks": result[0],
                    "high_res_multimasks": result[1],
                    "ious": result[2],
                    "low_res_masks": result[3],
                    "high_res_masks": result[4],
                    "obj_ptr": result[5],
                    "object_score_logits": result[6],
                },
            )
            recorder.add_record(
                stage="forward_sam_heads",
                frame_idx=frame_idx,
                metadata={
                    "multimask_output": bool(call_args.get("multimask_output", False)),
                    "has_point_inputs": call_args.get("point_inputs") is not None,
                    "has_mask_inputs": call_args.get("mask_inputs") is not None,
                    "has_gt_masks": call_args.get("gt_masks") is not None,
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_use_mask_as_output(self, *args, **kwargs):
        call_args = bind_call_arguments(original_use_mask_as_output, self, args, kwargs)
        frame_idx = recorder.current_context().get("frame_idx")
        result = original_use_mask_as_output(*args, **kwargs)
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {
                "backbone_features": call_args["backbone_features"],
                "mask_inputs": call_args["mask_inputs"],
            }
            if call_args.get("high_res_features") is not None:
                add_result_tree(
                    tensor_map, "high_res_features", call_args["high_res_features"]
                )
            add_result_tree(
                tensor_map,
                "use_mask_as_output",
                {
                    "low_res_multimasks": result[0],
                    "high_res_multimasks": result[1],
                    "ious": result[2],
                    "low_res_masks": result[3],
                    "high_res_masks": result[4],
                    "obj_ptr": result[5],
                    "object_score_logits": result[6],
                },
            )
            recorder.add_record(
                stage="use_mask_as_output",
                frame_idx=frame_idx,
                metadata={},
                tensors=tensor_map,
            )
        return result

    def wrapped_tracker_forward_image(self, *args, **kwargs):
        call_args = bind_call_arguments(original_tracker_forward_image, self, args, kwargs)
        context = recorder.current_context()
        frame_idx = context.get("frame_idx")
        result = original_tracker_forward_image(*args, **kwargs)
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {"image": call_args["img_batch"]}
            add_result_tree(tensor_map, "forward_image_output", result)
            recorder.add_record(
                stage="tracker_forward_image",
                frame_idx=frame_idx,
                metadata={
                    "context_stage": context.get("stage"),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_prompt_encoder_forward(self, *args, **kwargs):
        call_args = bind_call_arguments(original_prompt_encoder_forward, self, args, kwargs)
        frame_idx = recorder.current_context().get("frame_idx")
        result = original_prompt_encoder_forward(*args, **kwargs)
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {}
            add_result_tree(
                tensor_map,
                "prompt_encoder_inputs",
                {
                    "points": call_args.get("points"),
                    "boxes": call_args.get("boxes"),
                    "masks": call_args.get("masks"),
                },
            )
            add_result_tree(
                tensor_map,
                "prompt_encoder_output",
                {
                    "sparse_embeddings": result[0],
                    "dense_embeddings": result[1],
                },
            )
            recorder.add_record(
                stage="sam_prompt_encoder",
                frame_idx=frame_idx,
                metadata={
                    "has_points": call_args.get("points") is not None,
                    "has_boxes": call_args.get("boxes") is not None,
                    "has_masks": call_args.get("masks") is not None,
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_mask_decoder_forward(self, *args, **kwargs):
        call_args = bind_call_arguments(original_mask_decoder_forward, self, args, kwargs)
        frame_idx = recorder.current_context().get("frame_idx")
        result = original_mask_decoder_forward(*args, **kwargs)
        if frame_idx is not None and recorder.should_capture(frame_idx):
            tensor_map = {}
            add_result_tree(
                tensor_map,
                "mask_decoder_inputs",
                {
                    "image_embeddings": call_args["image_embeddings"],
                    "image_pe": call_args["image_pe"],
                    "sparse_prompt_embeddings": call_args["sparse_prompt_embeddings"],
                    "dense_prompt_embeddings": call_args["dense_prompt_embeddings"],
                    "high_res_features": call_args.get("high_res_features"),
                },
            )
            add_result_tree(
                tensor_map,
                "mask_decoder_output",
                {
                    "low_res_multimasks": result[0],
                    "ious": result[1],
                    "sam_output_tokens": result[2],
                    "object_score_logits": result[3],
                },
            )
            recorder.add_record(
                stage="sam_mask_decoder",
                frame_idx=frame_idx,
                metadata={
                    "multimask_output": bool(call_args.get("multimask_output", False)),
                    "repeat_image": bool(call_args.get("repeat_image", False)),
                },
                tensors=tensor_map,
            )
        return result

    def wrapped_encode_new_memory(self, *args, **kwargs):
        context = recorder.current_context()
        call_args = bind_call_arguments(original_encode_new_memory, self, args, kwargs)
        result = original_encode_new_memory(*args, **kwargs)
        frame_idx = context.get("frame_idx")
        if frame_idx is not None and recorder.should_capture(frame_idx):
            maskmem_features, maskmem_pos_enc = result
            tensor_map = {
                "pred_masks_high_res": call_args["pred_masks_high_res"],
                "object_score_logits": call_args["object_score_logits"],
                "maskmem_features": maskmem_features,
            }
            for idx, pos_enc in enumerate(maskmem_pos_enc):
                tensor_map[f"maskmem_pos_enc.{idx}"] = pos_enc
            recorder.add_record(
                stage="encode_new_memory",
                frame_idx=frame_idx,
                metadata={
                    "context_stage": context.get("stage"),
                    "is_mask_from_pts": bool(call_args["is_mask_from_pts"]),
                    "is_init_cond_frame": bool(call_args.get("is_init_cond_frame", False)),
                    "sigmoid_scale_for_mem_enc": float(self.sigmoid_scale_for_mem_enc),
                    "sigmoid_bias_for_mem_enc": float(self.sigmoid_bias_for_mem_enc),
                    "mask_for_mem_threshold": 0.0 if call_args["is_mask_from_pts"] and not self.training else None,
                    "mask_for_mem_mode": "binary_gt_0" if call_args["is_mask_from_pts"] and not self.training else "sigmoid_logits",
                },
                tensors=tensor_map,
            )
        return result

    if original_get_visual_prompt is not None:
        predictor_impl._get_visual_prompt = types.MethodType(
            wrapped_get_visual_prompt, predictor_impl
        )
    if original_get_image_feature is not None:
        predictor_impl._get_image_feature = types.MethodType(
            wrapped_get_image_feature, predictor_impl
        )
    if original_predictor_run_single_frame_inference is not None:
        predictor_impl._run_single_frame_inference = types.MethodType(
            wrapped_predictor_run_single_frame_inference, predictor_impl
        )
    if original_build_tracker_output is not None:
        predictor_impl._build_tracker_output = types.MethodType(
            wrapped_build_tracker_output, predictor_impl
        )
    if original_postprocess_output is not None:
        predictor_impl._postprocess_output = types.MethodType(
            wrapped_postprocess_output, predictor_impl
        )
    if original_preflight is not None:
        predictor_impl.propagate_in_video_preflight = types.MethodType(
            wrapped_preflight, predictor_impl
        )
    if original_clear_non_cond_mem is not None:
        predictor_impl._clear_non_cond_mem_around_input = types.MethodType(
            wrapped_clear_non_cond_mem, predictor_impl
        )
    if model is not None and original_tracker_add_new_objects is not None:
        model._tracker_add_new_objects = types.MethodType(wrapped_tracker_add_new_objects, model)
    if model is not None and original_det_track_one_frame is not None:
        model._det_track_one_frame = types.MethodType(
            wrapped_det_track_one_frame, model
        )
    tracker._run_memory_encoder = types.MethodType(wrapped_run_memory_encoder, tracker)
    tracker._prepare_memory_conditioned_features = types.MethodType(
        wrapped_prepare_memory_conditioned_features, tracker
    )
    tracker.track_step = types.MethodType(wrapped_track_step, tracker)
    tracker._encode_new_memory = types.MethodType(wrapped_encode_new_memory, tracker)
    if original_forward_sam_heads is not None:
        tracker._forward_sam_heads = types.MethodType(
            wrapped_forward_sam_heads, tracker
        )
    if original_use_mask_as_output is not None:
        tracker._use_mask_as_output = types.MethodType(
            wrapped_use_mask_as_output, tracker
        )
    if original_tracker_forward_image is not None:
        tracker.forward_image = types.MethodType(
            wrapped_tracker_forward_image, tracker
        )
    if original_prompt_encoder_forward is not None:
        tracker.sam_prompt_encoder.forward = types.MethodType(
            wrapped_prompt_encoder_forward, tracker.sam_prompt_encoder
        )
    if original_mask_decoder_forward is not None:
        tracker.sam_mask_decoder.forward = types.MethodType(
            wrapped_mask_decoder_forward, tracker.sam_mask_decoder
        )
    if original_add_new_points_or_box is not None:
        tracker.add_new_points_or_box = types.MethodType(
            wrapped_add_new_points_or_box, tracker
        )
    if original_add_new_mask is not None:
        tracker.add_new_mask = types.MethodType(wrapped_add_new_mask, tracker)
    return recorder


def mask_spec_to_tensor(mask_spec, frame_path: Path):
    from PIL import Image
    import numpy as np

    if not isinstance(mask_spec, dict):
        raise ValueError("mask prompt action requires a `mask` object")
    frame = Image.open(frame_path).convert("RGB")
    width, height = frame.size
    mask_type = mask_spec.get("type", "box_xyxy")
    if mask_type == "box_xyxy":
        box_xyxy = mask_spec.get("box_xyxy")
        if box_xyxy is None:
            raise ValueError("mask spec of type `box_xyxy` requires `box_xyxy`")
        mask = normalized_box_xyxy_to_mask(box_xyxy, width, height)
    elif mask_type == "image":
        mask_path = Path(mask_spec["path"]).expanduser().resolve()
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        if mask.shape != (height, width):
            mask = np.array(
                Image.fromarray((mask * 255.0).round().astype("uint8"), mode="L").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        threshold = float(mask_spec.get("threshold", 0.5))
        mask = (mask >= threshold).astype("uint8")
    else:
        raise ValueError(f"unsupported mask spec type {mask_type!r}")
    return torch.from_numpy(mask.astype("float32"))


def tracker_frame_object_score(inference_state, frame_idx, obj_id):
    obj_idx = inference_state["obj_id_to_idx"].get(int(obj_id))
    if obj_idx is None:
        return None
    candidate_maps = [
        inference_state.get("temp_output_dict_per_obj", {}).get(obj_idx, {}),
        inference_state.get("output_dict_per_obj", {}).get(obj_idx, {}),
    ]
    for mapping in candidate_maps:
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            frame_output = mapping.get(storage_key, {}).get(frame_idx)
            if frame_output is None:
                continue
            logits = frame_output.get("object_score_logits")
            if torch.is_tensor(logits):
                return float(torch.sigmoid(logits.reshape(-1)[0]).item())
    return None


def normalize_tracker_outputs(inference_state, frame_idx, obj_ids, video_res_masks):
    import numpy as np

    if torch.is_tensor(video_res_masks):
        masks = video_res_masks.detach().cpu().numpy()
    else:
        masks = np.asarray(video_res_masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    binary_masks = masks > 0
    out_obj_ids = [int(obj_id) for obj_id in obj_ids]
    out_binary_masks = [mask.astype("uint8") for mask in binary_masks]
    out_boxes_xywh = [binary_mask_to_box_xywh(mask) for mask in binary_masks]
    out_probs = [tracker_frame_object_score(inference_state, frame_idx, obj_id) for obj_id in out_obj_ids]
    return {
        "out_obj_ids": out_obj_ids,
        "out_probs": out_probs,
        "out_boxes_xywh": out_boxes_xywh,
        "out_binary_masks": out_binary_masks,
    }


def ensure_tracker_frame_features(tracker, inference_state, frame_idx, batch_size=1):
    get_image_feature = getattr(tracker, "_get_image_feature", None)
    if get_image_feature is None:
        return
    get_image_feature(inference_state, frame_idx, batch_size)


def execute_video_scenario_action_video_inference(
    predictor, session_id, action, frame_outputs, debug_records, debug_dir, args, default_prompt_text
):
    action_type = action["type"]
    if action_type == "add_prompt":
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(action["frame_idx"]),
        }
        if action.get("clear_old_points") is not None:
            request["clear_old_points"] = bool(action["clear_old_points"])
        if action.get("clear_old_boxes") is not None:
            request["clear_old_boxes"] = bool(action["clear_old_boxes"])
        if action.get("output_prob_thresh") is not None:
            request["output_prob_thresh"] = float(action["output_prob_thresh"])
        if action.get("text") is not None:
            request["text"] = action.get("text")
        if action.get("boxes_xywh") is not None:
            request["bounding_boxes"] = action["boxes_xywh"]
            request["bounding_box_labels"] = [
                int(label) for label in action.get("box_labels", [1 for _ in action["boxes_xywh"]])
            ]
        if action.get("points_xy_normalized") is not None:
            request["points"] = action["points_xy_normalized"]
            point_labels = action.get(
                "point_labels",
                [default_positive_label() for _ in action["points_xy_normalized"]],
            )
            request["point_labels"] = [int(label) for label in point_labels]
        if action.get("obj_id") is not None:
            request["obj_id"] = int(action["obj_id"])
        response = predictor.handle_request(request)
        merge_frame_outputs(
            frame_outputs,
            int(response["frame_index"]),
            response["outputs"],
        )
        if args.video_debug_bundle and should_capture_video_debug(args, int(response["frame_index"]), int(action["frame_idx"])):
            outputs = response.get("outputs", {})
            for obj_id, score, box_xywh, mask in zip(
                outputs.get("out_obj_ids", []),
                outputs.get("out_probs", []),
                outputs.get("out_boxes_xywh", []),
                outputs.get("out_binary_masks", []),
            ):
                if not should_capture_video_debug_obj(args, obj_id):
                    continue
                debug_records.append(
                    {
                        "stage": "prompt_frame_output",
                        "obj_id": int(obj_id),
                        "frame_idx": int(response["frame_index"]),
                        "prompt_frame_idx": int(action["frame_idx"]),
                        "prompt_metadata": build_video_debug_prompt_metadata_from_action(
                            action, default_prompt_text
                        ),
                        "observable": build_video_debug_observable(
                            debug_dir,
                            int(response["frame_index"]),
                            obj_id,
                            "prompt_frame_output_mask",
                            mask,
                            score,
                            box_xywh,
                        ),
                        "tracker_state": None,
                        "propagation_input": None,
                    }
                )
        return

    if action_type == "propagate":
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": action.get("direction", "forward"),
            "start_frame_index": action.get("start_frame_idx", None),
            "max_frame_num_to_track": action.get("max_frame_num_to_track", None),
        }
        for response in predictor.handle_stream_request(request):
            frame_idx = int(response["frame_index"])
            merge_frame_outputs(frame_outputs, frame_idx, response["outputs"])
            if not args.video_debug_bundle or not should_capture_video_debug(
                args, frame_idx, action.get("start_frame_idx", None)
            ):
                continue
            outputs = response.get("outputs", {})
            for obj_id, score, box_xywh, mask in zip(
                outputs.get("out_obj_ids", []),
                outputs.get("out_probs", []),
                outputs.get("out_boxes_xywh", []),
                outputs.get("out_binary_masks", []),
            ):
                if not should_capture_video_debug_obj(args, obj_id):
                    continue
                debug_records.append(
                    {
                        "stage": "propagated_output",
                        "obj_id": int(obj_id),
                        "frame_idx": frame_idx,
                        "prompt_frame_idx": int(action.get("start_frame_idx", 0) or 0),
                        "prompt_metadata": None,
                        "observable": build_video_debug_observable(
                            debug_dir,
                            frame_idx,
                            obj_id,
                            "propagated_mask",
                            mask,
                            score,
                            box_xywh,
                        ),
                        "tracker_state": None,
                        "propagation_input": None,
                    }
                )
        return

    raise ValueError(f"unsupported video-inference scenario action {action_type!r}")


def execute_video_scenario_action_tracker(
    tracker, inference_state, frame_paths, action, frame_outputs, debug_records, debug_dir, args
):
    action_type = action["type"]
    frame_idx = int(action.get("frame_idx", 0))
    if action_type == "add_prompt":
        obj_id = int(action["obj_id"])
        with torch.inference_mode():
            ensure_tracker_frame_features(tracker, inference_state, frame_idx)
            points = action.get("points_xy_normalized")
            point_labels = action.get(
                "point_labels",
                [default_positive_label() for _ in points] if points is not None else None,
            )
            box = action.get("box_xyxy")
            frame_idx_out, obj_ids, _, video_res_masks = tracker.add_new_points_or_box(
                inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=(torch.tensor(points, dtype=torch.float32) if points is not None else None),
                labels=(torch.tensor(point_labels, dtype=torch.int32) if point_labels is not None else None),
                clear_old_points=bool(action.get("clear_old_points", True)),
                rel_coordinates=bool(action.get("rel_coordinates", True)),
                use_prev_mem_frame=bool(action.get("use_prev_mem_frame", False)),
                box=(torch.tensor(box, dtype=torch.float32) if box is not None else None),
            )
        outputs = normalize_tracker_outputs(
            inference_state, int(frame_idx_out), obj_ids, video_res_masks
        )
        merge_frame_outputs(frame_outputs, int(frame_idx_out), outputs)
        if args.video_debug_bundle and should_capture_video_debug(args, int(frame_idx_out), frame_idx):
            for obj_id_out, score, box_xywh, mask in zip(
                outputs.get("out_obj_ids", []),
                outputs.get("out_probs", []),
                outputs.get("out_boxes_xywh", []),
                outputs.get("out_binary_masks", []),
            ):
                if not should_capture_video_debug_obj(args, obj_id_out):
                    continue
                debug_records.append(
                    {
                        "stage": "prompt_frame_output",
                        "obj_id": int(obj_id_out),
                        "frame_idx": int(frame_idx_out),
                        "prompt_frame_idx": frame_idx,
                        "prompt_metadata": build_video_debug_prompt_metadata_from_action(action, None),
                        "observable": build_video_debug_observable(
                            debug_dir,
                            int(frame_idx_out),
                            obj_id_out,
                            "prompt_frame_output_mask",
                            mask,
                            score,
                            box_xywh,
                        ),
                        "tracker_state": None,
                        "propagation_input": None,
                    }
                )
        return

    if action_type == "add_mask":
        obj_id = int(action["obj_id"])
        with torch.inference_mode():
            ensure_tracker_frame_features(tracker, inference_state, frame_idx)
            mask = mask_spec_to_tensor(action["mask"], frame_paths[frame_idx])
            frame_idx_out, obj_ids, _, video_res_masks = tracker.add_new_mask(
                inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
                add_mask_to_memory=bool(action.get("add_mask_to_memory", False)),
            )
        outputs = normalize_tracker_outputs(
            inference_state, int(frame_idx_out), obj_ids, video_res_masks
        )
        merge_frame_outputs(frame_outputs, int(frame_idx_out), outputs)
        if args.video_debug_bundle and should_capture_video_debug(args, int(frame_idx_out), frame_idx):
            for obj_id_out, score, box_xywh, mask in zip(
                outputs.get("out_obj_ids", []),
                outputs.get("out_probs", []),
                outputs.get("out_boxes_xywh", []),
                outputs.get("out_binary_masks", []),
            ):
                if not should_capture_video_debug_obj(args, obj_id_out):
                    continue
                debug_records.append(
                    {
                        "stage": "prompt_frame_output",
                        "obj_id": int(obj_id_out),
                        "frame_idx": int(frame_idx_out),
                        "prompt_frame_idx": frame_idx,
                        "prompt_metadata": build_video_debug_prompt_metadata_from_action(action, None),
                        "observable": build_video_debug_observable(
                            debug_dir,
                            int(frame_idx_out),
                            obj_id_out,
                            "prompt_frame_output_mask",
                            mask,
                            score,
                            box_xywh,
                        ),
                        "tracker_state": None,
                        "propagation_input": None,
                    }
                )
        return

    if action_type == "propagate":
        direction = action.get("direction", "forward")
        reverse_values = [False, True] if direction == "both" else [direction == "backward"]
        for reverse in reverse_values:
            with torch.inference_mode():
                propagation = tracker.propagate_in_video(
                    inference_state,
                    start_frame_idx=action.get("start_frame_idx", None),
                    max_frame_num_to_track=action.get("max_frame_num_to_track", None),
                    reverse=reverse,
                    propagate_preflight=True,
                )
                for frame_idx_out, obj_ids, _, video_res_masks, *_ in propagation:
                    outputs = normalize_tracker_outputs(
                        inference_state, int(frame_idx_out), obj_ids, video_res_masks
                    )
                merge_frame_outputs(frame_outputs, int(frame_idx_out), outputs)
                if not args.video_debug_bundle or not should_capture_video_debug(
                    args, int(frame_idx_out), action.get("start_frame_idx", None)
                ):
                    continue
                for obj_id_out, score, box_xywh, mask in zip(
                    outputs.get("out_obj_ids", []),
                    outputs.get("out_probs", []),
                    outputs.get("out_boxes_xywh", []),
                    outputs.get("out_binary_masks", []),
                ):
                    if not should_capture_video_debug_obj(args, obj_id_out):
                        continue
                    debug_records.append(
                        {
                            "stage": "propagated_output",
                            "obj_id": int(obj_id_out),
                            "frame_idx": int(frame_idx_out),
                            "prompt_frame_idx": int(action.get("start_frame_idx", 0) or 0),
                            "prompt_metadata": None,
                            "observable": build_video_debug_observable(
                                debug_dir,
                                int(frame_idx_out),
                                obj_id_out,
                                "propagated_mask",
                                mask,
                                score,
                                box_xywh,
                            ),
                            "tracker_state": None,
                            "propagation_input": None,
                        }
                    )
        return

    raise ValueError(f"unsupported tracker scenario action {action_type!r}")


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
        if args.video_scenario is not None and (
            args.prompt is not None or args.box or args.box_label
        ):
            raise ValueError(
                "`--video-scenario` drives video prompts internally; do not combine it with `--prompt`, `--box`, or `--box-label`"
            )
        if args.video_scenario is None and args.prompt is None and not args.box:
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
        debug_dir = output_dir / "debug"
        masks_dir.mkdir(parents=True, exist_ok=True)
        masked_frames_dir.mkdir(parents=True, exist_ok=True)
        if args.video_debug_bundle:
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = prepare_video_frames(
            source_video_path, frames_dir, max_frames=args.video_frame_count
        )
        resolved_box_labels = [
            int(label)
            for label in (args.box_label if args.box_label else [True for _ in args.box])
        ]
        scenario = (
            load_video_scenario(Path(args.video_scenario).expanduser().resolve())
            if args.video_scenario is not None
            else build_default_video_scenario(args, resolved_box_labels)
        )
        if args.video_debug_frame == [] and scenario.get("debug_capture_frame_indices"):
            args.video_debug_frame = list(scenario["debug_capture_frame_indices"])
        if args.video_debug_obj_id == [] and scenario.get("debug_capture_obj_ids"):
            args.video_debug_obj_id = list(scenario["debug_capture_obj_ids"])
        internal_recorder = None
        if args.video_debug_bundle:
            internal_capture_frame_indices = sorted(
                {
                    frame_idx
                    for frame_idx in (
                        scenario.get("debug_capture_frame_indices", [])
                        + list(range(min(4, len(frame_paths))))
                        + [int(frame_idx) for frame_idx in args.video_debug_frame]
                    )
                    if 0 <= frame_idx < len(frame_paths)
                }
            )
        frame_outputs = {}
        debug_records = []

        if scenario["engine"] == "video_inference":
            predictor = build_upstream_video_predictor_for_export(
                build_sam3_predictor=build_sam3_predictor,
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                apply_temporal_disambiguation=scenario["apply_temporal_disambiguation"],
                tracker_overrides=scenario["tracker_overrides"],
                predictor_overrides=scenario["predictor_overrides"],
            )
            if args.video_debug_bundle:
                internal_recorder = install_video_internal_fixture_recorder(
                    predictor,
                    debug_dir,
                    internal_capture_frame_indices,
                    scenario=scenario["raw"] or scenario,
                )
            response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frames_dir),
                    **scenario.get("session_overrides", {}),
                }
            )
            session_id = response["session_id"]
            if internal_recorder is not None:
                internal_recorder.set_session_id(session_id)
            for action in scenario["actions"]:
                execute_video_scenario_action_video_inference(
                    predictor,
                    session_id,
                    action,
                    frame_outputs,
                    debug_records,
                    debug_dir,
                    args,
                    default_prompt_text=effective_prompt,
                )
            active_model = predictor.model
            active_tracker = predictor.model.tracker
            engine_metadata = {"engine": "video_inference", "session_id": session_id}
        elif scenario["engine"] == "tracker":
            predictor = build_upstream_video_predictor_for_export(
                build_sam3_predictor=build_sam3_predictor,
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                apply_temporal_disambiguation=scenario["apply_temporal_disambiguation"],
                tracker_overrides=scenario["tracker_overrides"],
                predictor_overrides=scenario["predictor_overrides"],
            )
            tracker = predictor.model.tracker
            if getattr(tracker, "backbone", None) is None:
                tracker.backbone = predictor.model.detector.backbone
            if args.video_debug_bundle:
                internal_recorder = install_video_internal_fixture_recorder(
                    tracker,
                    debug_dir,
                    internal_capture_frame_indices,
                    scenario=scenario["raw"] or scenario,
                )
                internal_recorder.set_predictor_config(predictor.model, tracker)
            tracker_frames_dir = output_dir / "tracker_input_frames"
            prepare_tracker_input_frames(frame_paths, tracker_frames_dir)
            inference_state = tracker.init_state(
                video_path=str(tracker_frames_dir),
                async_loading_frames=False,
                **scenario.get("session_overrides", {}),
            )
            session_id = "tracker_direct"
            if internal_recorder is not None:
                internal_recorder.set_session_id(session_id)
            for action in scenario["actions"]:
                execute_video_scenario_action_tracker(
                    tracker,
                    inference_state,
                    frame_paths,
                    action,
                    frame_outputs,
                    debug_records,
                    debug_dir,
                    args,
                )
            active_model = predictor.model
            active_tracker = tracker
            engine_metadata = {"engine": "tracker", "session_id": session_id}
        else:
            raise ValueError(f"unsupported video scenario engine {scenario['engine']!r}")

        if internal_recorder is not None and active_model is None:
            internal_recorder.set_predictor_config(active_tracker, active_tracker)

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
                    prompt_text=args.prompt if args.video_scenario is None else None,
                    used_explicit_geometry=(
                        bool(args.box)
                        if args.video_scenario is None
                        else scenario_uses_explicit_geometry(scenario)
                    ),
                )
            )

        metadata = {
            "bundle_version": 1,
            "mode": "video_reference",
            "engine": scenario["engine"],
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
            "prompt_text": args.prompt if args.video_scenario is None else None,
            "points_xy_normalized": [],
            "point_labels": [],
            "boxes_cxcywh_normalized": [list(box) for box in args.box],
            "box_labels": resolved_box_labels if args.box else [],
            "frames_dir": "frames",
            "masks_dir": "masks",
            "masked_frames_dir": "masked_frames",
            "results_path": "video_results.json",
            "debug_dir": "debug" if args.video_debug_bundle else None,
            "video_apply_temporal_disambiguation": scenario["apply_temporal_disambiguation"],
            "scenario": scenario["raw"] or scenario,
            "tracker_config": capture_tracker_runtime_config(active_tracker),
            "predictor_config": (
                capture_predictor_runtime_config(active_model, active_tracker)
                if active_model is not None
                else capture_predictor_runtime_config(active_tracker, active_tracker)
            ),
            "checkpoint_path": str(checkpoint_path),
            "bpe_path": str(bpe_path),
            **engine_metadata,
        }
        with open(output_dir / "video_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        if args.video_debug_bundle:
            internal_manifest_path, internal_tensors_path = internal_recorder.finalize()
            debug_manifest = {
                "bundle_version": 1,
                "mode": "video_debug_bundle",
                "source": "upstream",
                "session_id": session_id,
                "internal_tracker_state_available": True,
                "capture_obj_ids": [int(obj_id) for obj_id in args.video_debug_obj_id]
                or scenario.get("debug_capture_obj_ids", []),
                "capture_frame_indices": [int(frame_idx) for frame_idx in args.video_debug_frame]
                or scenario.get("debug_capture_frame_indices", []),
                "internal_capture_frame_indices": list(internal_recorder.capture_frame_indices),
                "capture_first_propagated_only": False,
                "internal_manifest_path": str(internal_manifest_path.relative_to(debug_dir)),
                "internal_tensor_file": (
                    str(internal_tensors_path.relative_to(debug_dir))
                    if internal_tensors_path is not None
                    else None
                ),
                "scenario": scenario["raw"] or scenario,
                "records": debug_records,
            }
            with open(debug_dir / "debug_manifest.json", "w", encoding="utf-8") as f:
                json.dump(debug_manifest, f, indent=2)

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
        debug_tensors = {"vision.pre_block.patch_embed": to_cpu_nchw(x)}
        h, w = x.shape[1], x.shape[2]

        if trunk.pos_embed is not None:
            x = x + get_abs_pos(
                trunk.pos_embed,
                trunk.pretrain_use_cls_token,
                (h, w),
                trunk.retain_cls_token,
                tiling=trunk.tile_abs_pos,
            )
        debug_tensors["vision.pre_block.pos_embed_added"] = to_cpu_nchw(x)

        x = trunk.ln_pre(x)
        debug_tensors["vision.pre_block.ln_pre"] = to_cpu_nchw(x)

        if trunk.retain_cls_token:
            raise NotImplementedError("debug export does not support retained cls token")

        block_outputs = []
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

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode():
        with autocast_ctx:
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

        with autocast_ctx:
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
        "vision.weights.patch_embed.proj.weight": to_cpu_contiguous(
            model.backbone.vision_backbone.trunk.patch_embed.proj.weight
        ),
    }
    if model.backbone.vision_backbone.trunk.patch_embed.proj.bias is not None:
        tensors["vision.weights.patch_embed.proj.bias"] = to_cpu_contiguous(
            model.backbone.vision_backbone.trunk.patch_embed.proj.bias
        )
    if model.backbone.vision_backbone.trunk.pos_embed is not None:
        tensors["vision.weights.pos_embed"] = to_cpu_contiguous(
            model.backbone.vision_backbone.trunk.pos_embed
        )
    if hasattr(model.backbone.vision_backbone.trunk.ln_pre, "weight"):
        tensors["vision.weights.ln_pre.weight"] = to_cpu_contiguous(
            model.backbone.vision_backbone.trunk.ln_pre.weight
        )
    if hasattr(model.backbone.vision_backbone.trunk.ln_pre, "bias"):
        tensors["vision.weights.ln_pre.bias"] = to_cpu_contiguous(
            model.backbone.vision_backbone.trunk.ln_pre.bias
        )
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
