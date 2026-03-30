#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod parity;

use anyhow::{bail, Context, Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::sam3;
use image::{GrayImage, Luma, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

#[derive(Parser, Debug)]
struct Args {
    /// Optional path to the upstream `sam3.pt` checkpoint or a repo directory containing it.
    #[arg(long)]
    checkpoint: Option<String>,

    /// Optional path to a `tokenizer.json` or a repo directory containing it.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Optional image path for the vision and geometry smoke tests.
    #[arg(long)]
    image: Option<String>,

    /// Optional square resize used by the example smoke path before vision encoding.
    #[arg(long)]
    smoke_image_size: Option<usize>,

    /// Directory used for rendered overlay, mask, and summary outputs.
    #[arg(long, default_value = "candle-examples/examples/sam3/output")]
    output_dir: String,

    /// Optional parity bundle directory or `reference.safetensors` file.
    #[arg(long)]
    parity_bundle: Option<String>,

    /// Absolute tolerance used for stage-by-stage parity comparisons.
    #[arg(long, default_value_t = 1e-4f32)]
    parity_atol: f32,

    /// Optional text prompt for the text-encoder smoke test.
    #[arg(long)]
    prompt: Option<String>,

    /// Repeated normalized point prompts in `x,y` format.
    #[arg(long = "point", value_parser = parse_point)]
    points: Vec<PointArg>,

    /// Optional repeated point labels aligned with `--point`, defaults to `1`.
    #[arg(long = "point-label")]
    point_labels: Vec<u32>,

    /// Repeated normalized box prompts in `cx,cy,w,h` format.
    #[arg(long = "box", value_parser = parse_box)]
    boxes: Vec<BoxArg>,

    /// Optional repeated box labels aligned with `--box`, defaults to `1`.
    #[arg(long = "box-label")]
    box_labels: Vec<u32>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    print_config: bool,
}

#[derive(Clone, Copy, Debug)]
struct PointArg {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoxArg {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

fn parse_point(value: &str) -> std::result::Result<PointArg, String> {
    let coords = parse_floats(value, 2)?;
    Ok(PointArg {
        x: coords[0],
        y: coords[1],
    })
}

fn parse_box(value: &str) -> std::result::Result<BoxArg, String> {
    let coords = parse_floats(value, 4)?;
    Ok(BoxArg {
        cx: coords[0],
        cy: coords[1],
        w: coords[2],
        h: coords[3],
    })
}

fn parse_floats(value: &str, expected: usize) -> std::result::Result<Vec<f32>, String> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| format!("failed to parse `{value}` as comma-separated floats: {err}"))?;
    if parts.len() != expected {
        return Err(format!(
            "expected {expected} comma-separated values, got {} in `{value}`",
            parts.len()
        ));
    }
    Ok(parts)
}

fn resolve_repo_file(path: &str, expected_file: &str) -> std::path::PathBuf {
    let path = PathBuf::from(path);
    if path.is_dir() {
        path.join(expected_file)
    } else {
        path
    }
}

fn get_tokenizer(tokenizer: &str, context_length: usize) -> Result<Tokenizer> {
    let tokenizer_path = resolve_repo_file(tokenizer, "tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        E::msg(format!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        ))
    })?;
    let pad_id = *tokenizer
        .get_vocab(true)
        .get(CLIP_EOT_TOKEN)
        .ok_or_else(|| {
            E::msg(format!(
                "tokenizer is missing required token `{CLIP_EOT_TOKEN}`"
            ))
        })?;
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: CLIP_EOT_TOKEN.to_string(),
        }))
        .with_truncation(Some(TruncationParams {
            max_length: context_length,
            ..Default::default()
        }))
        .map_err(E::msg)?;
    Ok(tokenizer)
}

fn tokenize_prompt(
    prompt: &str,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let encoding = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let input_ids = Tensor::new(vec![encoding.get_ids().to_vec()], device)?;
    let attention_mask = Tensor::new(vec![encoding.get_attention_mask().to_vec()], device)?;
    Ok((input_ids, attention_mask))
}

fn preprocess_image_for_sam3(
    image_path: &str,
    image_size: usize,
    config: &sam3::Config,
    device: &Device,
) -> Result<Tensor> {
    let image = candle_examples::load_image_and_resize(image_path, image_size, image_size)?;
    let image = image.to_device(device)?;
    let mean = Tensor::from_vec(config.image.image_mean.to_vec(), (3, 1, 1), device)?;
    let std = Tensor::from_vec(config.image.image_std.to_vec(), (3, 1, 1), device)?;
    let image = (image.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?
        .unsqueeze(0)?;
    Ok(image)
}

fn load_render_image(image_path: &str, image_size: usize) -> Result<RgbaImage> {
    Ok(image::ImageReader::open(image_path)?
        .decode()
        .map_err(E::msg)?
        .resize_to_fill(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgba8())
}

fn best_query(scores: &Tensor) -> Result<(usize, f32)> {
    let scores = scores.to_vec3::<f32>()?;
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, score) in scores[0].iter().enumerate() {
        if score[0] > best_score {
            best_idx = idx;
            best_score = score[0];
        }
    }
    Ok((best_idx, best_score))
}

fn normalized_box_to_rect(box_xyxy: [f32; 4], image_size: usize) -> Rect {
    let scale = (image_size.saturating_sub(1)) as f32;
    let x0 = (box_xyxy[0].clamp(0.0, 1.0) * scale).round() as i32;
    let y0 = (box_xyxy[1].clamp(0.0, 1.0) * scale).round() as i32;
    let x1 = (box_xyxy[2].clamp(0.0, 1.0) * scale).round() as i32;
    let y1 = (box_xyxy[3].clamp(0.0, 1.0) * scale).round() as i32;
    let min_x = x0.min(x1);
    let min_y = y0.min(y1);
    let width = (x1.max(x0) - min_x).max(1) as u32;
    let height = (y1.max(y0) - min_y).max(1) as u32;
    Rect::at(min_x, min_y).of_size(width, height)
}

fn cxcywh_to_xyxy(bbox: &BoxArg) -> [f32; 4] {
    [
        bbox.cx - bbox.w * 0.5,
        bbox.cy - bbox.h * 0.5,
        bbox.cx + bbox.w * 0.5,
        bbox.cy + bbox.h * 0.5,
    ]
}

fn blend_mask(image: &mut RgbaImage, mask_probs: &[Vec<f32>], color: [u8; 3]) -> Result<GrayImage> {
    let height = mask_probs.len();
    let width = mask_probs.first().map(|row| row.len()).unwrap_or(0);
    let mut mask = GrayImage::new(width as u32, height as u32);
    for (y, row) in mask_probs.iter().enumerate() {
        for (x, prob) in row.iter().enumerate() {
            let prob = prob.clamp(0.0, 1.0);
            let mask_value = (prob * 255.0).round() as u8;
            mask.put_pixel(x as u32, y as u32, Luma([mask_value]));
            if prob >= 0.5 {
                let pixel = image.get_pixel_mut(x as u32, y as u32);
                let alpha = 0.35f32;
                pixel[0] = ((1.0 - alpha) * pixel[0] as f32 + alpha * color[0] as f32) as u8;
                pixel[1] = ((1.0 - alpha) * pixel[1] as f32 + alpha * color[1] as f32) as u8;
                pixel[2] = ((1.0 - alpha) * pixel[2] as f32 + alpha * color[2] as f32) as u8;
                pixel[3] = 255;
            }
        }
    }
    Ok(mask)
}

fn save_render_outputs(
    image_path: &str,
    image_size: usize,
    output_dir: &Path,
    prompt: &str,
    decoder: &sam3::DecoderOutput,
    segmentation: &sam3::SegmentationOutput,
    scores: &Tensor,
    input_boxes: &[BoxArg],
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let mut overlay = load_render_image(image_path, image_size)?;
    for bbox in input_boxes {
        draw_hollow_rect_mut(
            &mut overlay,
            normalized_box_to_rect(cxcywh_to_xyxy(bbox), image_size),
            Rgba([66, 135, 245, 255]),
        );
    }

    let (best_idx, best_score) = best_query(scores)?;
    let pred_boxes = decoder.pred_boxes_xyxy.to_vec3::<f32>()?;
    let best_box = pred_boxes[0][best_idx].clone();
    draw_hollow_rect_mut(
        &mut overlay,
        normalized_box_to_rect(
            [best_box[0], best_box[1], best_box[2], best_box[3]],
            image_size,
        ),
        Rgba([56, 201, 84, 255]),
    );

    let best_mask_logits = segmentation
        .mask_logits
        .i((0, best_idx))?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .upsample_bilinear2d(image_size, image_size, false)?
        .i((0, 0))?;
    let best_mask_probs = candle_nn::ops::sigmoid(&best_mask_logits)?;
    let best_mask_probs = best_mask_probs.to_vec2::<f32>()?;
    let mask = blend_mask(&mut overlay, &best_mask_probs, [56, 201, 84])?;

    let overlay_path = output_dir.join("overlay.png");
    let mask_path = output_dir.join("mask.png");
    overlay.save(&overlay_path)?;
    mask.save(&mask_path)?;

    let summary = json!({
        "prompt": prompt,
        "render_image_size": image_size,
        "best_query_index": best_idx,
        "best_score": best_score,
        "best_box_xyxy_normalized": best_box,
        "input_boxes_cxcywh_normalized": input_boxes.iter().map(|bbox| vec![bbox.cx, bbox.cy, bbox.w, bbox.h]).collect::<Vec<_>>(),
        "overlay_path": overlay_path.display().to_string(),
        "mask_path": mask_path.display().to_string(),
    });
    let summary_path = output_dir.join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("rendered outputs:");
    println!("  best query index: {best_idx}");
    println!("  best score: {best_score:.4}");
    println!("  best box xyxy (normalized): {:?}", best_box);
    println!("  overlay: {}", overlay_path.display());
    println!("  mask: {}", mask_path.display());
    println!("  summary: {}", summary_path.display());
    Ok(())
}

fn build_geometry_prompt(args: &Args, device: &Device) -> Result<Option<sam3::GeometryPrompt>> {
    if args.points.is_empty() && args.boxes.is_empty() {
        return Ok(None);
    }
    if !args.point_labels.is_empty() && args.point_labels.len() != args.points.len() {
        bail!(
            "`--point-label` count ({}) must match `--point` count ({})",
            args.point_labels.len(),
            args.points.len()
        )
    }
    if !args.box_labels.is_empty() && args.box_labels.len() != args.boxes.len() {
        bail!(
            "`--box-label` count ({}) must match `--box` count ({})",
            args.box_labels.len(),
            args.boxes.len()
        )
    }

    let points_xy = if args.points.is_empty() {
        None
    } else {
        let data = args
            .points
            .iter()
            .flat_map(|point| [point.x, point.y])
            .collect::<Vec<_>>();
        Some(Tensor::from_vec(data, (args.points.len(), 2), device)?)
    };
    let point_labels = if args.points.is_empty() {
        None
    } else {
        let labels = if args.point_labels.is_empty() {
            vec![1u32; args.points.len()]
        } else {
            args.point_labels.clone()
        };
        Some(Tensor::new(labels, device)?)
    };

    let boxes_cxcywh = if args.boxes.is_empty() {
        None
    } else {
        let data = args
            .boxes
            .iter()
            .flat_map(|bbox| [bbox.cx, bbox.cy, bbox.w, bbox.h])
            .collect::<Vec<_>>();
        Some(Tensor::from_vec(data, (args.boxes.len(), 4), device)?)
    };
    let box_labels = if args.boxes.is_empty() {
        None
    } else {
        let labels = if args.box_labels.is_empty() {
            vec![1u32; args.boxes.len()]
        } else {
            args.box_labels.clone()
        };
        Some(Tensor::new(labels, device)?)
    };

    Ok(Some(sam3::GeometryPrompt {
        boxes_cxcywh,
        box_labels,
        points_xy,
        point_labels,
        masks: None,
        mask_labels: None,
    }))
}

fn run_text_encoder(
    model: &sam3::Sam3ImageModel,
    prompt: &str,
    tokenizer_path: &str,
    context_length: usize,
    device: &Device,
) -> Result<sam3::TextEncoding> {
    let tokenizer = get_tokenizer(tokenizer_path, context_length)?;
    let (input_ids, attention_mask) = tokenize_prompt(prompt, &tokenizer, device)?;
    let encoding = model.encode_text_tokens(&input_ids, &attention_mask)?;
    println!("text stage:");
    println!("  text: {prompt}");
    println!("  input_ids: {:?}", input_ids.to_vec2::<u32>()?);
    println!("  attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);
    println!("  padding mask shape: {:?}", encoding.attention_mask.dims());
    println!(
        "  input embeddings shape: {:?}",
        encoding.input_embeddings.dims()
    );
    println!("  resized memory shape: {:?}", encoding.memory.dims());
    Ok(encoding)
}

fn run_vision_and_geometry(
    model: &sam3::Sam3ImageModel,
    image_path: &str,
    smoke_image_size: Option<usize>,
    output_dir: &Path,
    text_prompt: Option<&str>,
    text_encoding: Option<&sam3::TextEncoding>,
    geometry_prompt: Option<&sam3::GeometryPrompt>,
    input_boxes: &[BoxArg],
    device: &Device,
) -> Result<()> {
    let config = model.config();
    let (original_image, initial_h, initial_w) = candle_examples::load_image(image_path, None)?;
    let mut state = model.set_image(&original_image)?;
    if let Some(text_prompt) = text_prompt {
        state = state.with_text_prompt(text_prompt.to_string());
    }
    if let Some(geometry_prompt) = geometry_prompt {
        state = state.with_geometry_prompt(geometry_prompt.clone());
    }
    println!("typed image state:");
    println!("  original image size: {}x{}", initial_h, initial_w);
    println!(
        "  model input size: {}x{}",
        state.model_input_size.height, state.model_input_size.width
    );
    println!("  has text prompt: {}", state.text_prompt().is_some());
    println!(
        "  has geometry prompt: {}",
        !state.geometry_prompt().is_empty()
    );

    let image_size = smoke_image_size.unwrap_or(config.image.image_size);
    let image = preprocess_image_for_sam3(image_path, image_size, config, device)?;
    println!("vision stage:");
    println!("  preprocessed image shape: {:?}", image.dims());
    println!("  smoke resize: {image_size}x{image_size}");
    let visual = model.encode_image_features(&image)?;
    println!("  backbone_fpn levels: {}", visual.backbone_fpn.len());
    for (level_idx, (features, pos)) in visual
        .backbone_fpn
        .iter()
        .zip(visual.vision_pos_enc.iter())
        .enumerate()
    {
        println!(
            "  level {level_idx}: features {:?}, pos {:?}",
            features.dims(),
            pos.dims()
        );
    }
    println!(
        "  sam2 side neck present: {}",
        visual.sam2_backbone_fpn.is_some()
    );

    let empty_geometry = sam3::GeometryPrompt::default();
    let empty_encoded = model.encode_geometry_prompt(&empty_geometry, &visual)?;
    println!("geometry stage:");
    println!(
        "  empty prompt: features {:?}, padding mask {:?}",
        empty_encoded.features.dims(),
        empty_encoded.padding_mask.dims()
    );

    if let Some(geometry_prompt) = geometry_prompt {
        let encoded = model.encode_geometry_prompt(geometry_prompt, &visual)?;
        println!(
            "  user prompt: features {:?}, padding mask {:?}",
            encoded.features.dims(),
            encoded.padding_mask.dims()
        );
    }

    if let Some(text_encoding) = text_encoding {
        let fused = model.encode_fused_text(&visual, text_encoding)?;
        println!("fusion stage:");
        println!("  memory shape: {:?}", fused.memory.dims());
        println!("  pos embed shape: {:?}", fused.pos_embed.dims());
        println!("  padding mask shape: {:?}", fused.padding_mask.dims());
        println!(
            "  spatial shapes: {:?}",
            fused.spatial_shapes.to_vec2::<u32>()?
        );
        println!(
            "  level start index: {:?}",
            fused.level_start_index.to_vec1::<u32>()?
        );
        println!("  valid ratios shape: {:?}", fused.valid_ratios.dims());

        let decoder = model.decode_text_grounding(&fused, text_encoding)?;
        let scores = model.text_detection_scores(&decoder)?;
        println!("decoder stage:");
        println!("  queries shape: {:?}", decoder.queries.dims());
        println!("  pred logits shape: {:?}", decoder.pred_logits.dims());
        println!("  pred boxes shape: {:?}", decoder.pred_boxes.dims());
        println!(
            "  pred boxes xyxy shape: {:?}",
            decoder.pred_boxes_xyxy.dims()
        );
        println!("  text detection scores shape: {:?}", scores.dims());
        if let Some(presence_logits) = &decoder.presence_logits {
            println!("  presence logits shape: {:?}", presence_logits.dims());
        }

        let segmentation =
            model.segment_text_grounding(&visual, &decoder, &fused, text_encoding)?;
        println!("segmentation stage:");
        println!("  mask logits shape: {:?}", segmentation.mask_logits.dims());
        println!(
            "  semantic logits shape: {:?}",
            segmentation.semantic_logits.dims()
        );
        if let Some(presence_logits) = &segmentation.presence_logits {
            println!(
                "  segmentation presence logits shape: {:?}",
                presence_logits.dims()
            );
        }

        if let Some(text_prompt) = text_prompt {
            save_render_outputs(
                image_path,
                image_size,
                output_dir,
                text_prompt,
                &decoder,
                &segmentation,
                &scores,
                input_boxes,
            )?;
        }
    }

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let config = sam3::Config::default();
    let checkpoint_source = args
        .checkpoint
        .as_ref()
        .map(|path| sam3::Sam3CheckpointSource::upstream_pth(resolve_repo_file(path, "sam3.pt")));

    println!("sam3 example");
    println!("device: {device:?}");
    println!(
        "image MVP target: {}x{}",
        config.image.image_size, config.image.image_size
    );
    println!("milestones:");
    for step in sam3::Sam3ImageModel::scaffold_milestones() {
        println!("  - {step}");
    }

    if args.print_config {
        println!("{config:#?}");
    }

    if (args.image.is_some()
        || args.prompt.is_some()
        || !args.points.is_empty()
        || !args.boxes.is_empty()
        || args.parity_bundle.is_some())
        && checkpoint_source.is_none()
    {
        bail!("running implemented SAM3 stages currently requires `--checkpoint <sam3.pt>`")
    }
    if (!args.points.is_empty() || !args.boxes.is_empty()) && args.image.is_none() {
        bail!("`--point` and `--box` prompts require `--image` so the geometry encoder has image features")
    }
    if args.parity_bundle.is_some()
        && (args.image.is_some()
            || args.prompt.is_some()
            || args.tokenizer.is_some()
            || !args.points.is_empty()
            || !args.boxes.is_empty())
    {
        bail!(
            "`--parity-bundle` uses the exported reference inputs directly; omit `--image`, `--prompt`, `--tokenizer`, `--point`, and `--box`"
        )
    }

    let model = if let Some(checkpoint) = checkpoint_source.as_ref() {
        let model =
            sam3::Sam3ImageModel::from_checkpoint_source(&config, checkpoint, DType::F32, &device)?;
        println!("checkpoint opened and image-model namespace remap applied");
        Some(model)
    } else {
        None
    };

    if let Some(bundle_path) = args.parity_bundle.as_deref() {
        parity::run(
            model
                .as_ref()
                .context("SAM3 parity mode requires `--checkpoint <sam3.pt>`")?,
            &parity::ParityOptions {
                bundle_path: PathBuf::from(bundle_path),
                output_dir: PathBuf::from(&args.output_dir),
                atol: args.parity_atol,
            },
            &device,
        )?;
        return Ok(());
    }

    let geometry_prompt = build_geometry_prompt(&args, &device)?;

    let text_encoding = if let Some(prompt) = args.prompt.as_deref() {
        let tokenizer = args.tokenizer.as_deref().ok_or_else(|| {
            E::msg("encoding a SAM3 text prompt requires `--tokenizer <tokenizer.json>`")
        })?;
        Some(run_text_encoder(
            model
                .as_ref()
                .context("SAM3 text stage requires `--checkpoint <sam3.pt>`")?,
            prompt,
            tokenizer,
            config.text.context_length,
            &device,
        )?)
    } else {
        None
    };

    if let Some(image_path) = args.image.as_deref() {
        run_vision_and_geometry(
            model
                .as_ref()
                .context("SAM3 vision stage requires `--checkpoint <sam3.pt>`")?,
            image_path,
            args.smoke_image_size,
            Path::new(&args.output_dir),
            args.prompt.as_deref(),
            text_encoding.as_ref(),
            geometry_prompt.as_ref(),
            &args.boxes,
            &device,
        )?;
    }

    Ok(())
}
