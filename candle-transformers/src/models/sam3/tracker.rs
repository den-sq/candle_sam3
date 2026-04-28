use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use candle::{DType, Device, IndexOp, Result, Tensor, TensorId, D};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Embedding, LayerNorm, Module,
    VarBuilder,
};

use crate::models::segment_anything::{
    linear, prompt_encoder::PromptEncoder, transformer::TwoWayTransformer, LayerNorm2d, Linear,
};

use super::{
    checkpoint::Sam3CheckpointSource,
    neck::{Sam3DualViTDetNeck, VisualBackboneOutput},
    torch_ops::{
        interpolate::resize_bilinear2d_antialias,
        position::{build_2d_sine_position_encoding, get_1d_sine_pe},
    },
    vitdet::Sam3ViTDetTrunk,
    Config,
};

const NO_OBJ_SCORE: f64 = -1024.0;

mod config;
pub use config::*;
mod maskmem_backbone;
use maskmem_backbone::TrackerSimpleMaskEncoder;
mod memory_conditioning;
mod memory_transformer;
mod model;
use memory_transformer::TrackerMemoryTransformer;
mod prompt_inputs;
use prompt_inputs::normalize_mask_prompt;
mod sam_heads;
use sam_heads::{Sam3TrackerMaskDecoder, TrackerMlp};
#[derive(Debug, Clone)]
pub struct TrackerFrameState {
    pub low_res_masks: Tensor,
    pub high_res_masks: Tensor,
    pub iou_scores: Tensor,
    pub obj_ptr: Tensor,
    pub object_score_logits: Tensor,
    pub maskmem_features: Option<Tensor>,
    pub maskmem_pos_enc: Option<Tensor>,
    pub maskmem_prompt_features: Option<Tensor>,
    pub maskmem_prompt_pos_enc: Option<Tensor>,
    pub is_cond_frame: bool,
}

impl TrackerFrameState {
    pub fn set_maskmem_state(
        &mut self,
        maskmem_features: Tensor,
        maskmem_pos_enc: Tensor,
    ) -> Result<()> {
        let (maskmem_prompt_features, maskmem_prompt_pos_enc) =
            prepare_maskmem_prompt_tensors(&maskmem_features, &maskmem_pos_enc)?;
        self.maskmem_features = Some(maskmem_features);
        self.maskmem_pos_enc = Some(maskmem_pos_enc);
        self.maskmem_prompt_features = Some(maskmem_prompt_features);
        self.maskmem_prompt_pos_enc = Some(maskmem_prompt_pos_enc);
        Ok(())
    }

    pub fn clear_maskmem_state(&mut self) {
        self.maskmem_features = None;
        self.maskmem_pos_enc = None;
        self.maskmem_prompt_features = None;
        self.maskmem_prompt_pos_enc = None;
    }

    pub fn to_storage_device(&self, device: &candle::Device) -> Result<Self> {
        Ok(Self {
            low_res_masks: maybe_to_device(&self.low_res_masks, device)?,
            high_res_masks: maybe_to_device(&self.high_res_masks, device)?,
            iou_scores: maybe_to_device(&self.iou_scores, device)?,
            obj_ptr: maybe_to_device(&self.obj_ptr, device)?,
            object_score_logits: maybe_to_device(&self.object_score_logits, device)?,
            maskmem_features: self
                .maskmem_features
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, device))
                .transpose()?,
            maskmem_pos_enc: self
                .maskmem_pos_enc
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, device))
                .transpose()?,
            maskmem_prompt_features: self
                .maskmem_prompt_features
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, device))
                .transpose()?,
            maskmem_prompt_pos_enc: self
                .maskmem_prompt_pos_enc
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, device))
                .transpose()?,
            is_cond_frame: self.is_cond_frame,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PackedPromptHistory {
    initialized: bool,
    maskmem_frames: Vec<usize>,
    maskmem_frame_slots: HashMap<usize, usize>,
    maskmem_prompt_features: Option<Tensor>,
    maskmem_prompt_pos_enc: Option<Tensor>,
    obj_ptr_frames: Vec<usize>,
    obj_ptr_frame_slots: HashMap<usize, usize>,
    obj_ptrs: Option<Tensor>,
    maskmem_slot_tensor_cache: Arc<Mutex<HashMap<SlotTensorCacheKey, Tensor>>>,
    obj_ptr_slot_tensor_cache: Arc<Mutex<HashMap<SlotTensorCacheKey, Tensor>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SlotTensorCacheKey {
    device: String,
    frames: Vec<usize>,
}

impl Default for PackedPromptHistory {
    fn default() -> Self {
        Self {
            initialized: false,
            maskmem_frames: Vec::new(),
            maskmem_frame_slots: HashMap::new(),
            maskmem_prompt_features: None,
            maskmem_prompt_pos_enc: None,
            obj_ptr_frames: Vec::new(),
            obj_ptr_frame_slots: HashMap::new(),
            obj_ptrs: None,
            maskmem_slot_tensor_cache: Arc::new(Mutex::new(HashMap::new())),
            obj_ptr_slot_tensor_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl PackedPromptHistory {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn ensure_built(&mut self, states: &BTreeMap<usize, TrackerFrameState>) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        *self = Self::from_states(states)?;
        Ok(())
    }

    pub fn maskmem_prompt_features(&self) -> Option<&Tensor> {
        self.maskmem_prompt_features.as_ref()
    }

    pub fn maskmem_prompt_pos_enc(&self) -> Option<&Tensor> {
        self.maskmem_prompt_pos_enc.as_ref()
    }

    pub fn obj_ptrs(&self) -> Option<&Tensor> {
        self.obj_ptrs.as_ref()
    }

    pub fn maskmem_slot_indices_tensor(
        &self,
        frames: &[usize],
        device: &Device,
    ) -> Result<Option<Tensor>> {
        self.slot_indices_tensor(
            frames,
            &self.maskmem_frame_slots,
            &self.maskmem_slot_tensor_cache,
            device,
        )
    }

    pub fn obj_ptr_slot_indices_tensor(
        &self,
        frames: &[usize],
        device: &Device,
    ) -> Result<Option<Tensor>> {
        self.slot_indices_tensor(
            frames,
            &self.obj_ptr_frame_slots,
            &self.obj_ptr_slot_tensor_cache,
            device,
        )
    }

    pub fn append_state(&mut self, frame_idx: usize, state: &TrackerFrameState) -> Result<()> {
        if self.obj_ptr_frame_slots.contains_key(&frame_idx)
            || self.maskmem_frame_slots.contains_key(&frame_idx)
        {
            self.clear();
            return Ok(());
        }

        let next_obj_slot = self.obj_ptr_frames.len();
        let obj_ptr = state.obj_ptr.unsqueeze(0)?;
        Self::append_tensor(&mut self.obj_ptrs, obj_ptr)?;
        self.obj_ptr_frames.push(frame_idx);
        self.obj_ptr_frame_slots.insert(frame_idx, next_obj_slot);

        if let (Some(maskmem_prompt_features), Some(maskmem_prompt_pos_enc)) = (
            state.maskmem_prompt_features.as_ref(),
            state.maskmem_prompt_pos_enc.as_ref(),
        ) {
            let next_maskmem_slot = self.maskmem_frames.len();
            Self::append_tensor(
                &mut self.maskmem_prompt_features,
                maskmem_prompt_features.unsqueeze(0)?,
            )?;
            Self::append_tensor(
                &mut self.maskmem_prompt_pos_enc,
                maskmem_prompt_pos_enc.unsqueeze(0)?,
            )?;
            self.maskmem_frames.push(frame_idx);
            self.maskmem_frame_slots
                .insert(frame_idx, next_maskmem_slot);
        }

        self.initialized = true;
        Ok(())
    }

    fn from_states(states: &BTreeMap<usize, TrackerFrameState>) -> Result<Self> {
        let mut packed = Self::default();
        for (&frame_idx, state) in states.iter() {
            packed.append_state(frame_idx, state)?;
        }
        packed.initialized = true;
        Ok(packed)
    }

    fn append_tensor(dst: &mut Option<Tensor>, block: Tensor) -> Result<()> {
        let block = block.contiguous()?;
        *dst = Some(match dst.take() {
            Some(existing) => Tensor::cat(&[&existing, &block], 0)?,
            None => block,
        });
        Ok(())
    }

    fn slot_indices_tensor(
        &self,
        frames: &[usize],
        frame_slots: &HashMap<usize, usize>,
        cache: &Arc<Mutex<HashMap<SlotTensorCacheKey, Tensor>>>,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if frames.is_empty() {
            return Ok(None);
        }
        let key = SlotTensorCacheKey {
            device: format!("{:?}", device),
            frames: frames.to_vec(),
        };
        if let Some(cached) = cache
            .lock()
            .expect("slot tensor cache lock poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(Some(cached));
        }
        let mut slots = Vec::with_capacity(frames.len());
        for frame in frames {
            let Some(slot) = frame_slots.get(frame) else {
                return Ok(None);
            };
            slots.push(*slot as u32);
        }
        let slots = Tensor::from_vec(slots, frames.len(), device)?;
        let mut guard = cache.lock().expect("slot tensor cache lock poisoned");
        Ok(Some(
            guard.entry(key).or_insert_with(|| slots.clone()).clone(),
        ))
    }
}

pub(super) fn prepare_maskmem_prompt_tensors(
    maskmem_features: &Tensor,
    maskmem_pos_enc: &Tensor,
) -> Result<(Tensor, Tensor)> {
    Ok((
        maskmem_features.flatten(2, 3)?.permute((2, 0, 1))?,
        maskmem_pos_enc.flatten(2, 3)?.permute((2, 0, 1))?,
    ))
}

pub(super) fn maybe_to_device(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    if tensor.device().same_device(device) {
        Ok(tensor.clone())
    } else {
        tensor.to_device(device)
    }
}

pub(super) fn maybe_to_dtype(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        tensor.to_dtype(dtype)
    }
}

pub(super) fn maybe_to_device_dtype(
    tensor: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let tensor = maybe_to_device(tensor, device)?;
    maybe_to_dtype(&tensor, dtype)
}

#[derive(Debug, Clone)]
pub struct TrackerStepOutput {
    pub state: TrackerFrameState,
    pub prompt_frame_indices: Vec<usize>,
    pub memory_frame_indices: Vec<usize>,
}

#[cfg(feature = "sam3-parity-support")]
#[derive(Debug, Clone)]
pub struct ParityPreparedMemoryConditioning {
    pub pix_feat_with_mem: Tensor,
    pub selected_conditioning_frame_indices: Vec<usize>,
    pub selected_memory_frame_indices: Vec<usize>,
    pub selected_object_pointer_frame_indices: Vec<usize>,
}

#[cfg(feature = "sam3-parity-support")]
#[derive(Debug, Clone)]
pub struct ParityPreparedMemoryPrompt {
    pub prompt: Option<Tensor>,
    pub prompt_pos: Option<Tensor>,
    pub num_obj_ptr_tokens: usize,
    pub selected_conditioning_frame_indices: Vec<usize>,
    pub selected_memory_frame_indices: Vec<usize>,
    pub selected_object_pointer_frame_indices: Vec<usize>,
}

#[cfg(feature = "sam3-parity-support")]
pub trait Sam3TrackerParityExt {
    fn parity_compute_dtype(&self) -> DType;

    fn parity_prepare_high_res_features(&self, high_res_features: &[Tensor])
        -> Result<Vec<Tensor>>;

    fn parity_use_multimask(&self, is_init_cond_frame: bool, point_count: usize) -> bool;

    fn parity_get_tpos_enc(
        &self,
        rel_pos_list: &[i64],
        device: &Device,
        max_abs_pos: Option<usize>,
        dummy: bool,
    ) -> Result<Tensor>;

    fn parity_forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        point_prompt: Option<&(Tensor, Tensor)>,
        mask_inputs: Option<&Tensor>,
        high_res_features: Option<&[Tensor]>,
        multimask_output: bool,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState>;

    fn parity_use_mask_as_output(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<&[Tensor]>,
        mask_inputs: &Tensor,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState>;

    fn parity_prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &[Tensor],
        current_vision_pos_embeds: &[Tensor],
        feat_sizes: &[(usize, usize)],
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        use_prev_mem_frame: bool,
        packed_history: Option<&PackedPromptHistory>,
    ) -> Result<ParityPreparedMemoryConditioning>;

    fn parity_build_memory_conditioning_prompt(
        &self,
        frame_idx: usize,
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        packed_history: Option<&PackedPromptHistory>,
    ) -> Result<ParityPreparedMemoryPrompt>;

    fn parity_memory_transformer_forward(
        &self,
        src: &Tensor,
        prompt: &Tensor,
        src_pos: Option<&Tensor>,
        prompt_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor>;
}

#[derive(Debug)]
pub struct Sam3TrackerModel {
    config: Sam3TrackerConfig,
    vision_trunk: Option<Sam3ViTDetTrunk>,
    vision_neck: Option<Sam3DualViTDetNeck>,
    maskmem_backbone: TrackerSimpleMaskEncoder,
    memory_transformer: TrackerMemoryTransformer,
    mask_downsample: Conv2d,
    sam_prompt_encoder: PromptEncoder,
    sam_mask_decoder: Sam3TrackerMaskDecoder,
    obj_ptr_proj: TrackerMlp,
    obj_ptr_tpos_proj: Linear,
    maskmem_tpos_enc: Tensor,
    no_mem_embed: Tensor,
    no_mem_pos_enc: Tensor,
    no_obj_ptr: Tensor,
    no_obj_embed_spatial: Tensor,
    prepared_high_res_feature_cache: Mutex<HashMap<PreparedHighResFeatureCacheKey, Vec<Tensor>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PreparedHighResFeatureCacheKey {
    pub feat_s0_id: TensorId,
    pub feat_s1_id: TensorId,
    pub dtype: String,
}
