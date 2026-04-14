// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use candle::{Device, Result, Tensor};
use std::collections::HashMap;

use super::{geometry::GeometryPrompt, GroundingOutput, Sam3ImageModel};

/// Represents a tracked object across video frames
#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub obj_id: u32,
    pub object_masks: HashMap<usize, Tensor>, // frame_idx -> mask
    pub object_boxes: HashMap<usize, Tensor>, // frame_idx -> box
    pub object_scores: HashMap<usize, Tensor>, // frame_idx -> score
    pub creation_frame: usize,
    pub last_updated_frame: usize,
}

impl TrackedObject {
    pub fn new(obj_id: u32, creation_frame: usize) -> Self {
        Self {
            obj_id,
            object_masks: HashMap::new(),
            object_boxes: HashMap::new(),
            object_scores: HashMap::new(),
            creation_frame,
            last_updated_frame: creation_frame,
        }
    }

    pub fn update(&mut self, frame_idx: usize, mask: Tensor, box_: Tensor, score: Tensor) {
        self.object_masks.insert(frame_idx, mask);
        self.object_boxes.insert(frame_idx, box_);
        self.object_scores.insert(frame_idx, score);
        self.last_updated_frame = frame_idx;
    }
}

/// Per-frame inference state
#[derive(Debug)]
pub struct FrameInferenceState {
    pub frame_idx: usize,
    pub segmentation_output: Option<GroundingOutput>,
}

/// Video session state
#[derive(Debug)]
pub struct Sam3VideoSession {
    pub session_id: String,
    pub video_frames: Vec<Tensor>,
    pub frame_states: HashMap<usize, FrameInferenceState>,
    pub tracked_objects: HashMap<u32, TrackedObject>, // obj_id -> tracked object
    pub next_obj_id: u32,
    pub prompts_by_frame: HashMap<usize, SessionPrompt>,
}

#[derive(Debug, Clone)]
pub struct SessionPrompt {
    pub text: Option<String>,
    pub points: Option<Vec<(f32, f32)>>,
    pub point_labels: Option<Vec<u32>>,
    pub boxes: Option<Vec<(f32, f32, f32, f32)>>,
    pub box_labels: Option<Vec<u32>>,
}

impl Sam3VideoSession {
    pub fn new(session_id: String, video_frames: Vec<Tensor>) -> Self {
        Self {
            session_id,
            video_frames,
            frame_states: HashMap::new(),
            tracked_objects: HashMap::new(),
            next_obj_id: 0,
            prompts_by_frame: HashMap::new(),
        }
    }

    pub fn add_prompt(&mut self, frame_idx: usize, prompt: SessionPrompt) -> Result<()> {
        if frame_idx >= self.video_frames.len() {
            candle::bail!(
                "frame_idx {} exceeds video length {}",
                frame_idx,
                self.video_frames.len()
            );
        }
        self.prompts_by_frame.insert(frame_idx, prompt);
        Ok(())
    }

    pub fn add_tracked_object(&mut self, creation_frame: usize) -> u32 {
        let obj_id = self.next_obj_id;
        self.next_obj_id += 1;
        let tracked = TrackedObject::new(obj_id, creation_frame);
        self.tracked_objects.insert(obj_id, tracked);
        obj_id
    }

    pub fn remove_object(&mut self, obj_id: u32) -> Result<()> {
        self.tracked_objects.remove(&obj_id);
        Ok(())
    }

    pub fn num_frames(&self) -> usize {
        self.video_frames.len()
    }

    pub fn get_frame(&self, frame_idx: usize) -> Result<&Tensor> {
        self.video_frames
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} out of bounds", frame_idx)))
    }
}

/// Video prediction configuration
#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub score_threshold: f32,
    pub hotstart_delay: usize,
    pub max_objects: i32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            hotstart_delay: 0,
            max_objects: -1, // unlimited
        }
    }
}

/// Manages video sessions and performs frame propagation
pub struct Sam3VideoPredictor<'a> {
    model: &'a Sam3ImageModel,
    device: &'a Device,
    video_config: VideoConfig,
    sessions: HashMap<String, Sam3VideoSession>,
    next_session_id: usize,
}

impl<'a> Sam3VideoPredictor<'a> {
    pub fn new(model: &'a Sam3ImageModel, device: &'a Device) -> Self {
        Self {
            model,
            device,
            video_config: VideoConfig::default(),
            sessions: HashMap::new(),
            next_session_id: 0,
        }
    }

    pub fn with_config(mut self, config: VideoConfig) -> Self {
        self.video_config = config;
        self
    }

    /// Start a new video session
    pub fn start_session(&mut self, video_frames: Vec<Tensor>) -> Result<String> {
        let session_id = format!("session_{}", self.next_session_id);
        self.next_session_id += 1;

        let session = Sam3VideoSession::new(session_id.clone(), video_frames);
        self.sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Add a prompt to a frame in the session
    pub fn add_prompt(
        &mut self,
        session_id: &str,
        frame_idx: usize,
        prompt: SessionPrompt,
    ) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;

        session.add_prompt(frame_idx, prompt)?;
        Ok(())
    }

    /// Process a single frame and return the segmentation output
    pub fn process_frame(
        &self,
        session_id: &str,
        frame_idx: usize,
    ) -> Result<Option<GroundingOutput>> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;

        let image = session.get_frame(frame_idx)?;

        let prompt = match session.prompts_by_frame.get(&frame_idx) {
            Some(prompt) => prompt,
            None => return Ok(None),
        };

        if prompt.points.is_none() && prompt.boxes.is_none() {
            candle::bail!(
                "video frame prompts currently require points or boxes; text-only prompts are not yet implemented"
            );
        }

        let geometry_prompt = Self::session_prompt_to_geometry(prompt, self.device)?;
        let mut state = self.model.set_image(image)?;
        state = state.with_geometry_prompt(geometry_prompt);
        let grounding = self.model.ground_geometry(&state)?;
        Ok(Some(grounding))
    }

    fn session_prompt_to_geometry(
        prompt: &SessionPrompt,
        device: &Device,
    ) -> Result<GeometryPrompt> {
        let mut geometry_prompt = GeometryPrompt::default();

        if let Some(points) = prompt.points.as_ref() {
            let mut data = Vec::with_capacity(points.len() * 2);
            for (x, y) in points.iter() {
                data.push(*x);
                data.push(*y);
            }
            geometry_prompt.points_xy = Some(Tensor::from_vec(data, (points.len(), 2), device)?);
        }

        if let Some(point_labels) = prompt.point_labels.as_ref() {
            let labels = point_labels.iter().map(|v| *v as u32).collect::<Vec<_>>();
            geometry_prompt.point_labels =
                Some(Tensor::from_vec(labels, (point_labels.len(),), device)?);
        }

        if let Some(boxes) = prompt.boxes.as_ref() {
            let mut data = Vec::with_capacity(boxes.len() * 4);
            for (cx, cy, w, h) in boxes.iter() {
                data.push(*cx);
                data.push(*cy);
                data.push(*w);
                data.push(*h);
            }
            geometry_prompt.boxes_cxcywh = Some(Tensor::from_vec(data, (boxes.len(), 4), device)?);
        }

        if let Some(box_labels) = prompt.box_labels.as_ref() {
            geometry_prompt.box_labels = Some(Tensor::from_vec(
                box_labels.clone(),
                (box_labels.len(),),
                device,
            )?);
        }

        Ok(geometry_prompt)
    }

    /// Propagate tracked objects through video frames
    pub fn propagate_in_video(
        &self,
        session_id: &str,
        direction: PropagationDirection,
    ) -> Result<VideoOutput> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;

        let mut outputs_per_frame: HashMap<usize, GroundingOutput> = HashMap::new();

        // Find frames with prompts (seeds for propagation)
        let seed_frames: Vec<usize> = session.prompts_by_frame.keys().copied().collect::<Vec<_>>();
        if seed_frames.is_empty() {
            candle::bail!("no prompts added to session");
        }

        // Process seed frames first
        for frame_idx in &seed_frames {
            if let Some(output) = self.process_frame(session_id, *frame_idx)? {
                outputs_per_frame.insert(*frame_idx, output);
            }
        }

        // Forward propagation (from last seed frame to end)
        if matches!(
            direction,
            PropagationDirection::Forward | PropagationDirection::Both
        ) {
            if let Some(&last_seed) = seed_frames.iter().max() {
                for frame_idx in (last_seed + 1)..session.num_frames() {
                    // Propagate using optical flow / feature matching
                    // For now, use simple copy of previous frame's objects
                    if let Some(prev_output) = outputs_per_frame.get(&(frame_idx - 1)) {
                        outputs_per_frame.insert(frame_idx, prev_output.clone());
                    }
                }
            }
        }

        // Backward propagation (from first seed frame to start)
        if matches!(
            direction,
            PropagationDirection::Backward | PropagationDirection::Both
        ) {
            if let Some(&first_seed) = seed_frames.iter().min() {
                for frame_idx in (0..first_seed).rev() {
                    if let Some(next_output) = outputs_per_frame.get(&(frame_idx + 1)) {
                        outputs_per_frame.insert(frame_idx, next_output.clone());
                    }
                }
            }
        }

        Ok(VideoOutput { outputs_per_frame })
    }

    pub fn close_session(&mut self, session_id: &str) -> Result<()> {
        self.sessions.remove(session_id);
        Ok(())
    }

    pub fn reset_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.tracked_objects.clear();
            session.prompts_by_frame.clear();
        }
        Ok(())
    }

    /// Get the number of frames in a session
    pub fn session_frame_count(&self, session_id: &str) -> Result<usize> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        Ok(session.num_frames())
    }

    /// Get a frame from a session
    pub fn get_session_frame(&self, session_id: &str, frame_idx: usize) -> Result<Tensor> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        let frame = session.get_frame(frame_idx)?;
        Ok(frame.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropagationDirection {
    Forward,
    Backward,
    Both,
}

#[derive(Debug)]
pub struct VideoOutput {
    pub outputs_per_frame: HashMap<usize, GroundingOutput>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracked_object_creation() -> Result<()> {
        let obj = TrackedObject::new(0, 0);
        assert_eq!(obj.obj_id, 0);
        assert_eq!(obj.creation_frame, 0);
        Ok(())
    }

    #[test]
    fn test_video_session_creation() -> Result<()> {
        let device = Device::Cpu;
        let frame = Tensor::zeros((3, 1008, 1008), candle::DType::F32, &device)?;
        let session = Sam3VideoSession::new("test".to_string(), vec![frame]);
        assert_eq!(session.num_frames(), 1);
        Ok(())
    }
}
