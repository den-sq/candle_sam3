// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use anyhow::{bail, Context, Result};
use candle::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_transformers::models::sam3;
use image::Rgba;
use imageproc::drawing::draw_hollow_rect_mut;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Interactive refinement session for iterative mask improvement.
pub struct Sam3InteractiveSession<'a> {
    model: &'a sam3::Sam3ImageModel,
    device: Device,
    base_image_state: sam3::Sam3ImageState,
    image_state: sam3::Sam3ImageState,
    initial_state: Option<sam3::Sam3ImageState>,
    refinement_history: Vec<sam3::GroundingOutput>,
    current_mask: Option<Tensor>,
}

impl<'a> Sam3InteractiveSession<'a> {
    pub fn new(
        model: &'a sam3::Sam3ImageModel,
        device: Device,
        image_tensor: Tensor,
    ) -> CandleResult<Self> {
        let image_state = model.set_image(&image_tensor)?;
        Ok(Self {
            model,
            device,
            base_image_state: image_state.clone(),
            image_state,
            initial_state: None,
            refinement_history: Vec::new(),
            current_mask: None,
        })
    }

    /// Add the initial prompt and get the first prediction.
    pub fn initialize(
        &mut self,
        prompt: sam3::GeometryPrompt,
    ) -> CandleResult<&sam3::GroundingOutput> {
        self.image_state = self.base_image_state.clone().with_geometry_prompt(prompt);
        let output = self.model.ground_geometry(&self.image_state)?;
        self.image_state = self.image_state.clone().with_last_output(output.clone());
        self.initial_state = Some(self.image_state.clone());
        self.current_mask = Some(output.masks.clone());
        self.refinement_history.push(output);
        Ok(self.refinement_history.last().expect("history just pushed"))
    }

    /// Add refinement points and update the mask.
    pub fn refine(
        &mut self,
        additional_points: Vec<(f32, f32)>,
        point_labels: Vec<u32>,
    ) -> CandleResult<&sam3::GroundingOutput> {
        if !point_labels.is_empty() && point_labels.len() != additional_points.len() {
            candle::bail!(
                "interactive refinement expected {} point labels, got {}",
                additional_points.len(),
                point_labels.len()
            )
        }

        let existing_prompt = self.image_state.geometry_prompt().clone();
        let resolved_point_labels = if additional_points.is_empty() {
            Vec::new()
        } else if point_labels.is_empty() {
            vec![1; additional_points.len()]
        } else {
            point_labels
        };

        let new_points_xy = if additional_points.is_empty() {
            None
        } else {
            let data = additional_points
                .iter()
                .flat_map(|(x, y)| [*x, *y])
                .collect::<Vec<_>>();
            Some(Tensor::from_vec(
                data,
                (additional_points.len(), 2),
                &self.device,
            )?)
        };

        let new_point_labels = if additional_points.is_empty() {
            None
        } else {
            Some(Tensor::new(resolved_point_labels, &self.device)?)
        };

        let combined_points_xy = match (&existing_prompt.points_xy, &new_points_xy) {
            (Some(existing), Some(new)) => Some(Tensor::cat(&[existing, new], 0)?),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new)) => Some(new.clone()),
            (None, None) => None,
        };

        let combined_point_labels = match (&existing_prompt.point_labels, &new_point_labels) {
            (Some(existing), Some(new)) => Some(Tensor::cat(&[existing, new], 0)?),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new)) => Some(new.clone()),
            (None, None) => None,
        };

        let refined_prompt = sam3::GeometryPrompt {
            boxes_cxcywh: existing_prompt.boxes_cxcywh,
            box_labels: existing_prompt.box_labels,
            points_xy: combined_points_xy,
            point_labels: combined_point_labels,
            masks: existing_prompt.masks,
            mask_labels: existing_prompt.mask_labels,
        };

        self.image_state = self.image_state.clone().with_geometry_prompt(refined_prompt);
        let output = self.model.ground_geometry(&self.image_state)?;
        self.image_state = self.image_state.clone().with_last_output(output.clone());
        if self.initial_state.is_none() {
            self.initial_state = Some(self.image_state.clone());
        }
        self.current_mask = Some(output.masks.clone());
        self.refinement_history.push(output);
        Ok(self.refinement_history.last().expect("history just pushed"))
    }

    pub fn history(&self) -> &[sam3::GroundingOutput] {
        &self.refinement_history
    }

    /// Reset to the initial prompt state.
    pub fn reset(&mut self) -> CandleResult<()> {
        if let Some(initial_state) = self.initial_state.clone() {
            self.image_state = initial_state;
            if let Some(first_output) = self.refinement_history.first() {
                self.current_mask = Some(first_output.masks.clone());
                self.refinement_history.truncate(1);
            }
        } else {
            self.image_state = self.base_image_state.clone();
            self.current_mask = None;
            self.refinement_history.clear();
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InteractiveReplayStep {
    pub name: Option<String>,
    pub points: Vec<(f32, f32)>,
    pub point_labels: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum InteractiveReplayFile {
    Steps(Vec<InteractiveReplayStepFile>),
    Named { steps: Vec<InteractiveReplayStepFile> },
}

#[derive(Debug, Deserialize)]
struct InteractiveReplayStepFile {
    name: Option<String>,
    #[serde(default)]
    points: Vec<InteractiveReplayPointFile>,
}

#[derive(Debug, Deserialize)]
struct InteractiveReplayPointFile {
    x: f32,
    y: f32,
    #[serde(default = "default_positive_label")]
    label: u32,
}

#[derive(Debug, Clone)]
pub struct InteractiveMode {
    pub image_path: String,
    pub initial_points: Vec<(f32, f32)>,
    pub initial_point_labels: Vec<u32>,
    pub initial_boxes: Vec<(f32, f32, f32, f32)>,
    pub initial_box_labels: Vec<u32>,
    pub replay_steps: Vec<InteractiveReplayStep>,
    pub replay_script_path: Option<String>,
}

impl InteractiveMode {
    pub fn new(image_path: String) -> Self {
        Self {
            image_path,
            initial_points: Vec::new(),
            initial_point_labels: Vec::new(),
            initial_boxes: Vec::new(),
            initial_box_labels: Vec::new(),
            replay_steps: Vec::new(),
            replay_script_path: None,
        }
    }

    pub fn with_initial_points(mut self, points: Vec<(f32, f32)>, labels: Vec<u32>) -> Self {
        self.initial_points = points;
        self.initial_point_labels = labels;
        self
    }

    pub fn with_initial_boxes(
        mut self,
        boxes: Vec<(f32, f32, f32, f32)>,
        labels: Vec<u32>,
    ) -> Self {
        self.initial_boxes = boxes;
        self.initial_box_labels = labels;
        self
    }

    pub fn with_replay_steps(
        mut self,
        replay_steps: Vec<InteractiveReplayStep>,
        replay_script_path: Option<String>,
    ) -> Self {
        self.replay_steps = replay_steps;
        self.replay_script_path = replay_script_path;
        self
    }
}

#[derive(Debug, Serialize)]
struct InteractiveIterationSummary {
    iteration_index: usize,
    step_name: String,
    script_step_index: Option<usize>,
    added_points_xy_normalized: Vec<Vec<f32>>,
    added_point_labels: Vec<u32>,
    accumulated_points_xy_normalized: Vec<Vec<f32>>,
    accumulated_point_labels: Vec<u32>,
    initial_boxes_cxcywh_normalized: Vec<Vec<f32>>,
    initial_box_labels: Vec<u32>,
    best_score: f32,
    presence_score: Option<f32>,
    best_box_xyxy_normalized: Vec<f32>,
    render_image_size: RenderImageSize,
    base_path: String,
    overlay_path: String,
    mask_path: String,
}

#[derive(Debug, Serialize)]
struct InteractiveReplaySummary {
    image_path: String,
    replay_script_path: Option<String>,
    total_iterations: usize,
    iterations: Vec<InteractiveIterationSummary>,
}

#[derive(Debug, Serialize)]
struct RenderImageSize {
    width: usize,
    height: usize,
}

fn default_positive_label() -> u32 {
    1
}

pub fn load_replay_steps(path: &str) -> Result<Vec<InteractiveReplayStep>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read interactive replay manifest from {path}"))?;
    let manifest = serde_json::from_str::<InteractiveReplayFile>(&raw)
        .with_context(|| format!("failed to parse interactive replay manifest JSON from {path}"))?;
    let steps = match manifest {
        InteractiveReplayFile::Steps(steps) => steps,
        InteractiveReplayFile::Named { steps } => steps,
    };
    if steps.is_empty() {
        bail!("interactive replay manifest `{path}` does not contain any steps");
    }
    let parsed = steps
        .into_iter()
        .enumerate()
        .map(|(idx, step)| {
            if step.points.is_empty() {
                bail!(
                    "interactive replay step {} in `{path}` does not contain any points",
                    idx
                );
            }
            Ok(InteractiveReplayStep {
                name: step.name,
                points: step.points.iter().map(|point| (point.x, point.y)).collect(),
                point_labels: step.points.iter().map(|point| point.label).collect(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(parsed)
}

fn point_args(points: &[(f32, f32)]) -> Vec<crate::PointArg> {
    points
        .iter()
        .map(|(x, y)| crate::PointArg { x: *x, y: *y })
        .collect()
}

fn box_args(boxes: &[(f32, f32, f32, f32)]) -> Vec<crate::BoxArg> {
    boxes
        .iter()
        .map(|(cx, cy, w, h)| crate::BoxArg {
            cx: *cx,
            cy: *cy,
            w: *w,
            h: *h,
        })
        .collect()
}

fn interactive_geometry_inputs(
    points: &[(f32, f32)],
    point_labels: &[u32],
    boxes: &[(f32, f32, f32, f32)],
    box_labels: &[u32],
    device: &Device,
) -> Result<Option<crate::GeometryInputs>> {
    crate::build_geometry_prompt_from_parts(
        &point_args(points),
        point_labels,
        &box_args(boxes),
        box_labels,
        device,
    )
}

fn sanitize_step_name(step_name: &str) -> String {
    let sanitized = step_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        "step".to_string()
    } else {
        trimmed.to_string()
    }
}

fn iteration_dir(output_dir: &Path, iteration_index: usize, step_name: &str) -> PathBuf {
    output_dir.join(format!(
        "step_{iteration_index:03}_{}",
        sanitize_step_name(step_name)
    ))
}

fn first_f32_value(tensor: &Tensor, label: &str) -> Result<f32> {
    tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .next()
        .with_context(|| format!("interactive grounding output did not contain a {label}"))
}

fn render_iteration(
    image_path: &str,
    output_dir: &Path,
    iteration_index: usize,
    step_name: &str,
    script_step_index: Option<usize>,
    grounding: &sam3::GroundingOutput,
    accumulated_points: &[(f32, f32)],
    accumulated_point_labels: &[u32],
    step_points: &[(f32, f32)],
    step_point_labels: &[u32],
    initial_boxes: &[(f32, f32, f32, f32)],
    initial_box_labels: &[u32],
    image_size: usize,
) -> Result<InteractiveIterationSummary> {
    let iteration_dir = iteration_dir(output_dir, iteration_index, step_name);
    std::fs::create_dir_all(&iteration_dir)?;

    let base = crate::load_render_image(image_path)?;
    let render_width = base.width() as usize;
    let render_height = base.height() as usize;
    let base_path = iteration_dir.join("base.png");
    base.save(&base_path)?;

    let mut overlay = base.clone();
    crate::draw_prompt_annotations(
        &mut overlay,
        &point_args(accumulated_points),
        accumulated_point_labels,
        &box_args(initial_boxes),
        initial_box_labels,
        crate::RenderStyle::Combined,
    );

    let best_box = grounding
        .boxes_xyxy
        .to_vec2::<f32>()?
        .into_iter()
        .next()
        .context("interactive grounding output did not contain a predicted box")?;
    draw_hollow_rect_mut(
        &mut overlay,
        crate::normalized_box_to_rect(
            [best_box[0], best_box[1], best_box[2], best_box[3]],
            render_width,
            render_height,
        ),
        Rgba([56, 201, 84, 255]),
    );

    let best_mask_logits = match grounding.mask_logits.rank() {
        2 => grounding.mask_logits.clone(),
        3 => grounding.mask_logits.i(0)?,
        rank => bail!("interactive grounding mask logits expected rank 2 or 3, got {rank}"),
    };
    let mask_probs = crate::upsample_mask_probs_to_render(
        &best_mask_logits,
        image_size,
        image_path,
        crate::PreprocessMode::Exact,
    )?;
    let mask = crate::blend_mask(&mut overlay, &mask_probs, [56, 201, 84])?;

    let overlay_path = iteration_dir.join("overlay.png");
    let mask_path = iteration_dir.join("mask.png");
    overlay.save(&overlay_path)?;
    mask.save(&mask_path)?;

    let best_score = first_f32_value(&grounding.scores, "score")?;
    let presence_score = grounding
        .presence_scores
        .as_ref()
        .map(|tensor| first_f32_value(tensor, "presence score"))
        .transpose()?;

    let summary = InteractiveIterationSummary {
        iteration_index,
        step_name: step_name.to_string(),
        script_step_index,
        added_points_xy_normalized: step_points
            .iter()
            .map(|(x, y)| vec![*x, *y])
            .collect(),
        added_point_labels: step_point_labels.to_vec(),
        accumulated_points_xy_normalized: accumulated_points
            .iter()
            .map(|(x, y)| vec![*x, *y])
            .collect(),
        accumulated_point_labels: accumulated_point_labels.to_vec(),
        initial_boxes_cxcywh_normalized: initial_boxes
            .iter()
            .map(|(cx, cy, w, h)| vec![*cx, *cy, *w, *h])
            .collect(),
        initial_box_labels: initial_box_labels.to_vec(),
        best_score,
        presence_score,
        best_box_xyxy_normalized: best_box,
        render_image_size: RenderImageSize {
            width: render_width,
            height: render_height,
        },
        base_path: base_path.display().to_string(),
        overlay_path: overlay_path.display().to_string(),
        mask_path: mask_path.display().to_string(),
    };

    let summary_path = iteration_dir.join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(summary)
}

pub fn run_interactive_refinement(
    model: &sam3::Sam3ImageModel,
    interactive_mode: &InteractiveMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let image_tensor =
        crate::preprocess_image_path_exact(&interactive_mode.image_path, model, device)?;
    let mut session = Sam3InteractiveSession::new(model, device.clone(), image_tensor)?;

    let initial_geometry_inputs = interactive_geometry_inputs(
        &interactive_mode.initial_points,
        &interactive_mode.initial_point_labels,
        &interactive_mode.initial_boxes,
        &interactive_mode.initial_box_labels,
        device,
    )?;

    if initial_geometry_inputs.is_none() && interactive_mode.replay_steps.is_empty() {
        bail!(
            "interactive mode requires initial prompts via --point/--box or a replay manifest via --interactive-script"
        );
    }

    std::fs::create_dir_all(output_dir)?;

    let mut summaries = Vec::new();
    let mut iteration_index = 0usize;
    let mut accumulated_points: Vec<(f32, f32)> = initial_geometry_inputs
        .as_ref()
        .map(|inputs| {
            inputs
                .points
                .iter()
                .map(|point| (point.x, point.y))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut accumulated_point_labels = initial_geometry_inputs
        .as_ref()
        .map(|inputs| inputs.point_labels.clone())
        .unwrap_or_default();
    let initial_boxes: Vec<(f32, f32, f32, f32)> = initial_geometry_inputs
        .as_ref()
        .map(|inputs| {
            inputs
                .boxes
                .iter()
                .map(|bbox| (bbox.cx, bbox.cy, bbox.w, bbox.h))
                .collect()
        })
        .unwrap_or_default();
    let initial_box_labels = initial_geometry_inputs
        .as_ref()
        .map(|inputs| inputs.box_labels.clone())
        .unwrap_or_default();

    if let Some(initial_inputs) = initial_geometry_inputs {
        let initial_output = session.initialize(initial_inputs.prompt)?;
        summaries.push(render_iteration(
            &interactive_mode.image_path,
            output_dir,
            iteration_index,
            "initial_prompt",
            None,
            initial_output,
            &accumulated_points,
            &accumulated_point_labels,
            &accumulated_points,
            &accumulated_point_labels,
            &initial_boxes,
            &initial_box_labels,
            model.config().image.image_size,
        )?);
        iteration_index += 1;
    }

    for (script_step_index, step) in interactive_mode.replay_steps.iter().enumerate() {
        let step_points = step.points.clone();
        let step_point_labels = step.point_labels.clone();
        let grounding = if session.history().is_empty() {
            let step_inputs =
                interactive_geometry_inputs(&step_points, &step_point_labels, &[], &[], device)?
                    .context("interactive replay step unexpectedly produced no geometry prompt")?;
            accumulated_points = step_inputs
                .points
                .iter()
                .map(|point| (point.x, point.y))
                .collect::<Vec<_>>();
            accumulated_point_labels = step_inputs.point_labels.clone();
            session.initialize(step_inputs.prompt)?
        } else {
            accumulated_points.extend(step_points.iter().copied());
            accumulated_point_labels.extend(step_point_labels.iter().copied());
            session.refine(step_points.clone(), step_point_labels.clone())?
        };

        let step_name = step
            .name
            .clone()
            .unwrap_or_else(|| format!("refinement_{script_step_index:02}"));
        summaries.push(render_iteration(
            &interactive_mode.image_path,
            output_dir,
            iteration_index,
            &step_name,
            Some(script_step_index),
            grounding,
            &accumulated_points,
            &accumulated_point_labels,
            &step_points,
            &step_point_labels,
            &initial_boxes,
            &initial_box_labels,
            model.config().image.image_size,
        )?);
        iteration_index += 1;
    }

    let replay_summary = InteractiveReplaySummary {
        image_path: interactive_mode.image_path.clone(),
        replay_script_path: interactive_mode.replay_script_path.clone(),
        total_iterations: summaries.len(),
        iterations: summaries,
    };
    let summary_path = output_dir.join("interactive_session.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&replay_summary)?)?;

    println!("interactive refinement complete");
    println!("  image: {}", interactive_mode.image_path);
    println!("  iterations: {}", replay_summary.total_iterations);
    println!("  session summary: {}", summary_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_replay_steps, run_interactive_refinement, InteractiveMode};
    use anyhow::Result;
    use candle::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_transformers::models::sam3::{
        Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
        Sam3ImageModel, SegmentationConfig, TextConfig, VisionConfig,
    };

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "sam3_interactive_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("current time should be after unix epoch")
                .as_nanos()
        ))
    }

    fn tiny_segmentation_config() -> Config {
        Config {
            image: ImageConfig {
                image_size: 56,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
            },
            vision: VisionConfig {
                image_size: 56,
                pretrain_image_size: 28,
                patch_size: 14,
                embed_dim: 32,
                depth: 0,
                num_heads: 4,
                mlp_ratio: 4.0,
                window_size: 2,
                global_attn_blocks: vec![],
                use_abs_pos: true,
                tile_abs_pos: true,
                use_rope: true,
                use_interp_rope: true,
                rope_theta: 10_000.0,
                rope_pt_size: 24,
                retain_cls_token: false,
                ln_pre: false,
            },
            text: TextConfig {
                d_model: 8,
                width: 16,
                heads: 2,
                layers: 1,
                context_length: 4,
                vocab_size: 16,
            },
            neck: NeckConfig {
                d_model: 8,
                scale_factors: [4.0, 2.0, 1.0, 0.5],
                scalp: 1,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 8,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 16,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 8,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 16,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 8,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 16,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 8,
                upsampling_stages: 3,
                aux_masks: false,
                presence_head: false,
            },
        }
    }

    fn checked_in_manifest_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/sam3/interactive_replay.example.json")
    }

    #[test]
    fn replay_manifest_defaults_positive_labels() -> Result<()> {
        let dir = unique_temp_dir("manifest");
        std::fs::create_dir_all(&dir)?;
        let manifest_path = dir.join("interactive.json");
        std::fs::write(
            &manifest_path,
            r#"{
  "steps": [
    { "name": "seed", "points": [{ "x": 0.48, "y": 0.50 }] },
    {
      "name": "trim",
      "points": [
        { "x": 0.12, "y": 0.15, "label": 0 },
        { "x": 0.55, "y": 0.72 }
      ]
    },
    { "points": [{ "x": 0.60, "y": 0.78, "label": 1 }] }
  ]
}"#,
        )?;

        let steps = load_replay_steps(manifest_path.to_str().expect("utf8 temp path"))?;
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].name.as_deref(), Some("seed"));
        assert_eq!(steps[0].point_labels, vec![1]);
        assert_eq!(steps[1].point_labels, vec![0, 1]);
        assert_eq!(steps[2].point_labels, vec![1]);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn checked_in_replay_manifest_has_three_steps() -> Result<()> {
        let manifest_path = checked_in_manifest_path();
        let steps = load_replay_steps(manifest_path.to_str().expect("utf8 manifest path"))?;
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].name.as_deref(), Some("seed_positive"));
        Ok(())
    }

    #[test]
    fn interactive_replay_writes_iteration_artifacts() -> Result<()> {
        let dir = unique_temp_dir("runner");
        std::fs::create_dir_all(&dir)?;

        let image_path = dir.join("fixture.png");
        let fixture = image::RgbImage::from_fn(32, 24, |x, y| {
            image::Rgb([
                ((x * 7) % 255) as u8,
                ((y * 11) % 255) as u8,
                (((x + y) * 13) % 255) as u8,
            ])
        });
        fixture.save(&image_path)?;

        let manifest_path = checked_in_manifest_path();
        let replay_steps = load_replay_steps(manifest_path.to_str().expect("utf8 manifest path"))?;
        let output_dir = dir.join("output");
        let device = Device::Cpu;
        let model = Sam3ImageModel::new(
            &tiny_segmentation_config(),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        let interactive_mode = InteractiveMode::new(
            image_path.to_str().expect("utf8 fixture path").to_string(),
        )
        .with_replay_steps(
            replay_steps,
            Some(manifest_path.display().to_string()),
        );

        run_interactive_refinement(&model, &interactive_mode, &output_dir, &device)?;

        let session_summary_path = output_dir.join("interactive_session.json");
        assert!(session_summary_path.exists());

        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&session_summary_path)?)?;
        assert_eq!(summary["total_iterations"].as_u64(), Some(3));

        let iterations = summary["iterations"]
            .as_array()
            .expect("session summary should include iterations");
        assert_eq!(iterations.len(), 3);
        for iteration in iterations {
            for key in ["base_path", "overlay_path", "mask_path"] {
                let path = std::path::PathBuf::from(
                    iteration[key]
                        .as_str()
                        .expect("artifact path should be a string"),
                );
                assert!(path.exists(), "expected artifact {} to exist", path.display());
            }
        }

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
