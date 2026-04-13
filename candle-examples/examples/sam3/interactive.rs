// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use candle::{Device, Result, Tensor};
use candle_transformers::models::sam3;
use std::path::Path;

/// Interactive refinement session for iterative mask improvement
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
    pub fn new(model: &'a sam3::Sam3ImageModel, device: Device, image_tensor: Tensor) -> Result<Self> {
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

    /// Add initial prompt and get first prediction
    pub fn initialize(&mut self, prompt: sam3::GeometryPrompt) -> Result<&sam3::GroundingOutput> {
        self.image_state = self.base_image_state.clone().with_geometry_prompt(prompt);
        let output = self.model.ground_geometry(&self.image_state)?;
        self.image_state = self.image_state.clone().with_last_output(output.clone());
        self.initial_state = Some(self.image_state.clone());
        self.current_mask = Some(output.masks.clone());
        self.refinement_history.push(output);
        Ok(self.refinement_history.last().unwrap())
    }

    /// Add refinement points and update mask
    pub fn refine(&mut self, additional_points: Vec<(f32, f32)>, point_labels: Vec<u32>) -> Result<&sam3::GroundingOutput> {
        if !point_labels.is_empty() && point_labels.len() != additional_points.len() {
            candle::bail!(
                "interactive refinement expected {} point labels, got {}",
                additional_points.len(),
                point_labels.len()
            )
        }

        // Combine existing geometry with new points
        let existing_prompt = self.image_state.geometry_prompt().clone();
        let resolved_point_labels = if additional_points.is_empty() {
            Vec::new()
        } else if point_labels.is_empty() {
            vec![1; additional_points.len()]
        } else {
            point_labels
        };

        // Convert new points to tensors
        let new_points_xy = if additional_points.is_empty() {
            None
        } else {
            let data: Vec<f32> = additional_points.iter().flat_map(|(x, y)| vec![*x, *y]).collect();
            Some(Tensor::from_vec(data, (additional_points.len(), 2), &self.device)?)
        };

        let new_point_labels = if additional_points.is_empty() {
            None
        } else {
            Some(Tensor::new(resolved_point_labels, &self.device)?)
        };

        // Append new points to existing ones
        let combined_points_xy = match (&existing_prompt.points_xy, &new_points_xy) {
            (Some(existing), Some(new)) => Some(Tensor::cat(&[existing, &new], 0)?),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new)) => Some(new.clone()),
            (None, None) => None,
        };

        let combined_point_labels = match (&existing_prompt.point_labels, &new_point_labels) {
            (Some(existing), Some(new)) => Some(Tensor::cat(&[existing, &new], 0)?),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new)) => Some(new.clone()),
            (None, None) => None,
        };

        // Create refined prompt
        let refined_prompt = sam3::GeometryPrompt {
            boxes_cxcywh: existing_prompt.boxes_cxcywh,
            box_labels: existing_prompt.box_labels,
            points_xy: combined_points_xy,
            point_labels: combined_point_labels,
            masks: existing_prompt.masks,
            mask_labels: existing_prompt.mask_labels,
        };

        // Update state and ground
        self.image_state = self.image_state.clone().with_geometry_prompt(refined_prompt);
        let output = self.model.ground_geometry(&self.image_state)?;
        self.image_state = self.image_state.clone().with_last_output(output.clone());
        if self.initial_state.is_none() {
            self.initial_state = Some(self.image_state.clone());
        }
        self.current_mask = Some(output.masks.clone());
        self.refinement_history.push(output);
        Ok(self.refinement_history.last().unwrap())
    }

    /// Get current mask
    pub fn current_mask(&self) -> Option<&Tensor> {
        self.current_mask.as_ref()
    }

    /// Get refinement history
    pub fn history(&self) -> &[sam3::GroundingOutput] {
        &self.refinement_history
    }

    /// Reset to initial state
    pub fn reset(&mut self) -> Result<()> {
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

/// Interactive refinement mode configuration
pub struct InteractiveMode {
    pub image_path: String,
    pub initial_points: Vec<(f32, f32)>,
    pub initial_point_labels: Vec<u32>,
    pub max_refinements: usize,
}

impl InteractiveMode {
    pub fn new(image_path: String) -> Self {
        Self {
            image_path,
            initial_points: Vec::new(),
            initial_point_labels: Vec::new(),
            max_refinements: 5,
        }
    }

    pub fn with_initial_points(mut self, points: Vec<(f32, f32)>, labels: Vec<u32>) -> Self {
        self.initial_points = points;
        self.initial_point_labels = labels;
        self
    }
}

/// Run interactive refinement mode
pub fn run_interactive_refinement(
    model: &sam3::Sam3ImageModel,
    interactive_mode: &InteractiveMode,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    // Load image using the parity-validated exact preprocessing path.
    let image_tensor = crate::preprocess_image_path_exact(
        &interactive_mode.image_path,
        model,
        device,
    )?;

    // Create interactive session
    let mut session = Sam3InteractiveSession::new(model, device.clone(), image_tensor)?;

    // Create initial prompt from points
    let initial_prompt = if !interactive_mode.initial_points.is_empty() {
        let points_xy = Tensor::from_vec(
            interactive_mode.initial_points.iter().flat_map(|(x, y)| vec![*x, *y]).collect(),
            (interactive_mode.initial_points.len(), 2),
            device,
        )?;
        let point_labels = Tensor::new(interactive_mode.initial_point_labels.clone(), device)?;

        sam3::GeometryPrompt {
            points_xy: Some(points_xy),
            point_labels: Some(point_labels),
            boxes_cxcywh: None,
            box_labels: None,
            masks: None,
            mask_labels: None,
        }
    } else {
        // Default empty prompt - would need user input in real interactive mode
        sam3::GeometryPrompt::default()
    };

    // Initialize with first prediction
    if !initial_prompt.is_empty() {
        let _initial_output = session.initialize(initial_prompt)?;
        println!("Interactive session initialized with {} initial points", interactive_mode.initial_points.len());
    } else {
        println!("Interactive session initialized - awaiting user input for refinement");
    }

    // In a real implementation, this would be an interactive loop:
    // 1. Display current mask
    // 2. Accept user clicks (positive/negative points)
    // 3. Refine mask
    // 4. Repeat until user is satisfied

    println!("Interactive refinement scaffolding complete");
    println!("Output directory: {}", output_dir.display());

    Ok(())
}
