use super::*;

impl Sam3TrackerModel {
    pub fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let model_device = vb.device().clone();
        let model_dtype = vb.dtype();
        let (vision_trunk, vision_neck) = if config.predictor.with_backbone {
            let vision_trunk = Some(Sam3ViTDetTrunk::new(
                &Config::default().vision,
                vb.pp("backbone").pp("vision_backbone").pp("trunk"),
            )?);
            let vision_neck = Some(Sam3DualViTDetNeck::new(
                &Config::default().neck,
                vb.pp("backbone").pp("vision_backbone"),
            )?);
            if let Some(vision_neck) = vision_neck.as_ref() {
                let default_vision = Config::default().vision;
                vision_neck.prime_position_encoding_cache(
                    &model_device,
                    model_dtype,
                    default_vision.image_size / default_vision.patch_size,
                    default_vision.image_size / default_vision.patch_size,
                )?;
            }
            (
                vision_trunk,
                vision_neck,
            )
        } else {
            (None, None)
        };
        let mask_downsample = candle_nn::conv2d(
            1,
            1,
            4,
            Conv2dConfig {
                stride: 4,
                ..Default::default()
            },
            vb.pp("mask_downsample"),
        )?;
        let maskmem_backbone = TrackerSimpleMaskEncoder::new(
            &config.maskmem_backbone,
            config.hidden_dim,
            vb.pp("maskmem_backbone"),
        )?;
        let memory_transformer =
            TrackerMemoryTransformer::new(&config.transformer, vb.pp("transformer").pp("encoder"))?;
        let sam_prompt_encoder = PromptEncoder::new(
            config.prompt_encoder.embed_dim,
            (
                config.prompt_encoder.image_embedding_size[0],
                config.prompt_encoder.image_embedding_size[1],
            ),
            (
                config.prompt_encoder.input_image_size[0],
                config.prompt_encoder.input_image_size[1],
            ),
            config.prompt_encoder.mask_in_chans,
            vb.pp("sam_prompt_encoder"),
        )?;
        let sam_mask_decoder =
            Sam3TrackerMaskDecoder::new(&config.mask_decoder, vb.pp("sam_mask_decoder"))?;
        let obj_ptr_proj = TrackerMlp::new(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            3,
            false,
            vb.pp("obj_ptr_proj"),
        )?;
        let obj_ptr_tpos_proj = linear(
            vb.pp("obj_ptr_tpos_proj"),
            config.hidden_dim,
            config.memory_dim,
            true,
        )?;
        Ok(Self {
            config: config.clone(),
            vision_trunk,
            vision_neck,
            maskmem_backbone,
            memory_transformer,
            mask_downsample,
            sam_prompt_encoder,
            sam_mask_decoder,
            obj_ptr_proj,
            obj_ptr_tpos_proj,
            maskmem_tpos_enc: vb.get(&config.shapes.maskmem_tpos_enc_shape, "maskmem_tpos_enc")?,
            no_mem_embed: vb.get(&config.shapes.no_mem_embed_shape, "no_mem_embed")?,
            no_mem_pos_enc: vb.get(&config.shapes.no_mem_pos_enc_shape, "no_mem_pos_enc")?,
            no_obj_ptr: vb.get(&config.shapes.no_obj_ptr_shape, "no_obj_ptr")?,
            no_obj_embed_spatial: vb.get(
                &config.shapes.no_obj_embed_spatial_shape,
                "no_obj_embed_spatial",
            )?,
            prepared_high_res_feature_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn from_checkpoint_source(
        sam3_config: &Config,
        checkpoint: &Sam3CheckpointSource,
        dtype: DType,
        device: &candle::Device,
    ) -> Result<Self> {
        let tracker_config = Sam3TrackerConfig::from_sam3_config(sam3_config);
        Self::new(
            &tracker_config,
            checkpoint.load_tracker_var_builder(dtype, device)?,
        )
    }

    pub fn config(&self) -> &Sam3TrackerConfig {
        &self.config
    }

    pub fn image_embedding_size(&self) -> usize {
        self.config.image_embedding_size()
    }

    pub fn low_res_mask_size(&self) -> usize {
        self.config.low_res_mask_size()
    }

    pub fn input_mask_size(&self) -> usize {
        self.config.shapes.input_mask_size
    }

    pub(super) fn get_tpos_enc(
        &self,
        rel_pos_list: &[i64],
        device: &Device,
        max_abs_pos: Option<usize>,
        dummy: bool,
    ) -> Result<Tensor> {
        if dummy {
            return Tensor::zeros(
                (rel_pos_list.len(), self.config.memory_dim),
                DType::F32,
                device,
            );
        }

        let t_diff_max = max_abs_pos
            .map(|value| value.saturating_sub(1).max(1))
            .unwrap_or(1) as f64;
        let pos_inds = device_f32_vector(device, rel_pos_list)?;
        let pos_inds = pos_inds.affine(1.0 / t_diff_max, 0.0)?;
        let pos_enc = get_1d_sine_pe(&pos_inds, self.config.hidden_dim)?;
        self.obj_ptr_tpos_proj.forward(&pos_enc)
    }

    pub fn encode_image_features(&self, image: &Tensor) -> Result<VisualBackboneOutput> {
        let vision_trunk = self.vision_trunk.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "tracker image-feature path is unavailable because predictor.with_backbone=false"
                    .to_owned(),
            )
        })?;
        let vision_neck = self.vision_neck.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "tracker image-feature path is unavailable because predictor.with_backbone=false"
                    .to_owned(),
            )
        })?;
        let image = match image.rank() {
            3 => image.unsqueeze(0)?,
            4 => image.clone(),
            rank => {
                candle::bail!(
                    "sam3 tracker image encoder expects CHW or BCHW input, got rank {rank}"
                )
            }
        };
        let trunk = vision_trunk.forward(&image)?;
        vision_neck.forward(&trunk)
    }

    pub fn track_frame(
        &self,
        visual: &VisualBackboneOutput,
        frame_idx: usize,
        num_frames: usize,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes_xyxy: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        history: &BTreeMap<usize, TrackerFrameState>,
        is_conditioning_frame: bool,
        reverse: bool,
        use_prev_mem_frame: bool,
        run_mem_encoder: bool,
    ) -> Result<TrackerStepOutput> {
        self.track_frame_with_storage_device(
            visual,
            frame_idx,
            num_frames,
            point_coords,
            point_labels,
            boxes_xyxy,
            mask_input,
            history,
            is_conditioning_frame,
            reverse,
            use_prev_mem_frame,
            run_mem_encoder,
            None,
            None,
        )
    }

    pub fn track_frame_with_storage_device(
        &self,
        visual: &VisualBackboneOutput,
        frame_idx: usize,
        num_frames: usize,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes_xyxy: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        history: &BTreeMap<usize, TrackerFrameState>,
        is_conditioning_frame: bool,
        reverse: bool,
        use_prev_mem_frame: bool,
        run_mem_encoder: bool,
        storage_device: Option<&Device>,
        packed_history: Option<&PackedPromptHistory>,
    ) -> Result<TrackerStepOutput> {
        if visual.backbone_fpn.is_empty() {
            candle::bail!("tracker requires at least one visual feature level")
        }
        if visual.vision_pos_enc.is_empty() {
            candle::bail!("tracker requires at least one visual position-encoding level")
        }
        let compute_dtype = self.no_obj_ptr.dtype();
        let prepared_high_res_features =
            if visual.backbone_fpn.len() > 1 {
                Some(self.prepare_high_res_features(
                    &visual.backbone_fpn[..visual.backbone_fpn.len() - 1],
                )?)
            } else {
                None
            };
        let high_res_features = prepared_high_res_features.as_deref();
        let mut prompt_frame_indices = if is_conditioning_frame {
            vec![frame_idx]
        } else {
            Vec::new()
        };
        let mut memory_frame_indices = Vec::new();
        let backbone_features = if !history.is_empty() {
            let (feat_sizes, current_vision_feats, current_vision_pos_embeds) =
                if let Some(sequences) = visual.tracker_sequences.as_ref() {
                    let current_vision_feats = sequences
                        .vision_feats
                        .iter()
                        .map(|feat| maybe_to_dtype(feat, compute_dtype))
                        .collect::<Result<Vec<_>>>()?;
                    let current_vision_pos_embeds = sequences
                        .vision_pos_embeds
                        .iter()
                        .map(|pos| maybe_to_dtype(pos, compute_dtype))
                        .collect::<Result<Vec<_>>>()?;
                    (
                        sequences.feat_sizes.clone(),
                        current_vision_feats,
                        current_vision_pos_embeds,
                    )
                } else {
                    let feat_sizes = visual
                        .backbone_fpn
                        .iter()
                        .zip(visual.vision_pos_enc.iter())
                        .map(|(feat, pos)| {
                            let (_, feat_channels, feat_h, feat_w) = feat.dims4()?;
                            let pos_shape = pos.dims4()?;
                            if pos_shape != (1, feat_channels, feat_h, feat_w) {
                                candle::bail!(
                                    "tracker expected matching feature/pos shapes, got ({feat_channels}, {feat_h}, {feat_w}) and {pos_shape:?}"
                                );
                            }
                            Ok((feat_h, feat_w))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let current_vision_feats = visual
                        .backbone_fpn
                        .iter()
                        .map(|feat| {
                            feat.to_dtype(compute_dtype)?
                                .permute((2, 3, 0, 1))?
                                .reshape((feat.dim(2)? * feat.dim(3)?, feat.dim(0)?, feat.dim(1)?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let current_vision_pos_embeds = visual
                        .vision_pos_enc
                        .iter()
                        .map(|pos| {
                            pos.to_dtype(compute_dtype)?
                                .permute((2, 3, 0, 1))?
                                .reshape((pos.dim(2)? * pos.dim(3)?, pos.dim(0)?, pos.dim(1)?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    (feat_sizes, current_vision_feats, current_vision_pos_embeds)
                };
            let prepared = self.prepare_memory_conditioned_features(
                frame_idx,
                is_conditioning_frame,
                current_vision_feats.as_slice(),
                current_vision_pos_embeds.as_slice(),
                feat_sizes.as_slice(),
                history,
                num_frames,
                reverse,
                use_prev_mem_frame,
                packed_history,
            )?;
            prompt_frame_indices = prepared.selected_conditioning_frame_indices;
            memory_frame_indices = prepared.selected_memory_frame_indices;
            maybe_to_dtype(&prepared.pix_feat_with_mem, compute_dtype)?
        } else {
            maybe_to_dtype(
                visual.backbone_fpn.last().expect("checked non-empty above"),
                compute_dtype,
            )?
        };
        if let Some(mask_input) = mask_input {
            let mask_input = normalize_mask_prompt(mask_input, backbone_features.device())?;
            let mask_inputs_float = mask_input.to_dtype(DType::F32)?;
            let high_res_masks = mask_inputs_float.affine(20.0, -10.0)?;
            let mask_input_low_res_size = (self.input_mask_size() / self.config.backbone_stride) * 4;
            let low_res_masks = resize_bilinear2d_antialias(
                &high_res_masks,
                mask_input_low_res_size,
                mask_input_low_res_size,
            )?;
            let iou_scores =
                Tensor::ones((mask_inputs_float.dim(0)?, 1), DType::F32, backbone_features.device())?;
            let mask_prompt = self.mask_downsample.forward(&mask_inputs_float)?;
            let mut state = self.use_mask_as_output_prepared(
                &backbone_features,
                high_res_features,
                mask_inputs_float,
                high_res_masks,
                low_res_masks,
                iou_scores,
                &mask_prompt,
                is_conditioning_frame,
            )?;
            if run_mem_encoder && self.config.num_maskmem > 0 {
                let (maskmem_features, maskmem_pos_enc) = self.encode_new_memory_from_visual(
                    visual,
                    &state.high_res_masks,
                    &state.object_score_logits,
                    false,
                )?;
                state.set_maskmem_state(maskmem_features, maskmem_pos_enc)?;
                state = self.maybe_offload_state_for_eval(state, storage_device)?;
            }
            return Ok(TrackerStepOutput {
                state,
                prompt_frame_indices,
                memory_frame_indices,
            });
        }
        let point_prompt = self.prepare_point_prompt(
            point_coords,
            point_labels,
            boxes_xyxy,
            backbone_features.device(),
        )?;
        let point_count = point_prompt
            .as_ref()
            .map(|(_, labels)| labels.dim(1).unwrap_or(0))
            .unwrap_or(0);
        let multimask_output = self.use_multimask(is_conditioning_frame, point_count);
        let mut state = self.forward_sam_heads(
            &backbone_features,
            point_prompt.as_ref(),
            None,
            high_res_features,
            multimask_output,
            is_conditioning_frame,
        )?;
        if run_mem_encoder && self.config.num_maskmem > 0 {
            let (maskmem_features, maskmem_pos_enc) = self.encode_new_memory_from_visual(
                visual,
                &state.high_res_masks,
                &state.object_score_logits,
                point_prompt.is_some(),
            )?;
            state.set_maskmem_state(maskmem_features, maskmem_pos_enc)?;
            state = self.maybe_offload_state_for_eval(state, storage_device)?;
        }
        Ok(TrackerStepOutput {
            state,
            prompt_frame_indices,
            memory_frame_indices,
        })
    }

    pub(super) fn apply_non_overlapping_constraints(&self, pred_masks: &Tensor) -> Result<Tensor> {
        let (batch_size, channels, height, width) = pred_masks.dims4()?;
        if batch_size == 1 {
            return Ok(pred_masks.clone());
        }
        let device = pred_masks.device();
        let max_obj_inds = pred_masks.argmax_keepdim(0)?;
        let batch_obj_inds =
            Tensor::arange(0u32, batch_size as u32, device)?.reshape((batch_size, 1, 1, 1))?;
        let keep = batch_obj_inds
            .broadcast_eq(&max_obj_inds.broadcast_as((batch_size, channels, height, width))?)?;
        let neg_ten = Tensor::full(-10f32, pred_masks.shape(), device)?;
        let suppressed = pred_masks.le(-10f64)?.where_cond(pred_masks, &neg_ten)?;
        keep.where_cond(pred_masks, &suppressed)
    }

    fn encode_new_memory_from_pix_feat(
        &self,
        pix_feat: &Tensor,
        pred_masks_high_res: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<(Tensor, Tensor)> {
        let pix_feat = maybe_to_dtype(pix_feat, self.no_obj_ptr.dtype())?;
        let mut pred_masks_high_res =
            normalize_mask_prompt(pred_masks_high_res, pix_feat.device())?;
        let object_score_logits = maybe_to_device_dtype(
            object_score_logits,
            pix_feat.device(),
            self.no_obj_ptr.dtype(),
        )?;
        if self.config.non_overlap_masks_for_mem_enc {
            pred_masks_high_res = self.apply_non_overlapping_constraints(&pred_masks_high_res)?;
        }
        let mut mask_for_mem = if is_mask_from_pts {
            pred_masks_high_res.gt(0f64)?.to_dtype(DType::F32)?
        } else {
            candle_nn::ops::sigmoid(&pred_masks_high_res)?
        };
        if self.config.sigmoid_scale_for_mem_enc != 1.0
            || self.config.sigmoid_bias_for_mem_enc != 0.0
        {
            mask_for_mem = mask_for_mem.affine(
                self.config.sigmoid_scale_for_mem_enc as f64,
                self.config.sigmoid_bias_for_mem_enc as f64,
            )?;
        }
        let (mut maskmem_features, maskmem_pos_enc) =
            self.maskmem_backbone
                .forward(&pix_feat, &mask_for_mem, true)?;
        let no_obj_scale = object_score_logits
            .le(0f64)?
            .to_dtype(maskmem_features.dtype())?;
        let no_obj_embed = maybe_to_device_dtype(
            &self.no_obj_embed_spatial,
            maskmem_features.device(),
            maskmem_features.dtype(),
        )?
        .reshape((1, self.config.memory_dim, 1, 1))?;
        let no_obj_add = no_obj_scale
            .reshape((no_obj_scale.dim(0)?, 1, 1, 1))?
            .broadcast_mul(&no_obj_embed)?;
        maskmem_features = maskmem_features.broadcast_add(&no_obj_add)?;
        Ok((maskmem_features, maskmem_pos_enc))
    }

    fn encode_new_memory_from_visual(
        &self,
        visual: &VisualBackboneOutput,
        pred_masks_high_res: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<(Tensor, Tensor)> {
        let pix_feat = visual.backbone_fpn.last().ok_or_else(|| {
            candle::Error::Msg(
                "tracker memory encoder requires a top-level backbone feature".into(),
            )
        })?;
        self.encode_new_memory_from_pix_feat(
            pix_feat,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
        )
    }

    pub(super) fn maybe_offload_state_for_eval(
        &self,
        state: TrackerFrameState,
        storage_device: Option<&Device>,
    ) -> Result<TrackerFrameState> {
        if !self.config.predictor.offload_output_to_cpu_for_eval {
            return Ok(state);
        }
        let storage = &Device::Cpu;
        if storage_device.is_some_and(|device| !device.same_device(storage)) {
            return Ok(state);
        }
        Ok(TrackerFrameState {
            low_res_masks: maybe_to_device(&state.low_res_masks, storage)?,
            high_res_masks: maybe_to_device(&state.high_res_masks, storage)?,
            iou_scores: maybe_to_device(&state.iou_scores, storage)?,
            obj_ptr: state.obj_ptr,
            object_score_logits: state.object_score_logits,
            maskmem_features: state
                .maskmem_features
                .as_ref()
                .map(|tensor| maybe_to_device(&maybe_to_dtype(tensor, DType::BF16)?, storage))
                .transpose()?,
            maskmem_pos_enc: state
                .maskmem_pos_enc
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, storage))
                .transpose()?,
            maskmem_prompt_features: state
                .maskmem_prompt_features
                .as_ref()
                .map(|tensor| maybe_to_device(&maybe_to_dtype(tensor, DType::BF16)?, storage))
                .transpose()?,
            maskmem_prompt_pos_enc: state
                .maskmem_prompt_pos_enc
                .as_ref()
                .map(|tensor| maybe_to_device(tensor, storage))
                .transpose()?,
            is_cond_frame: state.is_cond_frame,
        })
    }

    pub fn encode_state_memory(
        &self,
        visual: &VisualBackboneOutput,
        state: &TrackerFrameState,
    ) -> Result<(Tensor, Tensor)> {
        self.encode_new_memory_from_visual(
            visual,
            &state.high_res_masks,
            &state.object_score_logits,
            false,
        )
    }

    pub fn encode_external_memory(
        &self,
        visual: &VisualBackboneOutput,
        high_res_masks: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_points: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.encode_new_memory_from_visual(
            visual,
            high_res_masks,
            object_score_logits,
            is_mask_from_points,
        )
    }
}

fn device_f32_vector(device: &Device, values: &[i64]) -> Result<Tensor> {
    if values.is_empty() {
        return Tensor::zeros(0, DType::F32, device);
    }
    let parts = values
        .iter()
        .map(|value| Tensor::arange(*value as f32, *value as f32 + 1.0, device))
        .collect::<Result<Vec<_>>>()?;
    let part_refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(part_refs.as_slice(), 0)
}
