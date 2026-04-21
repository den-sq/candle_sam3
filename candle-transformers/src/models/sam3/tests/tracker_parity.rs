    fn tracker_build_config_matches_upstream_contract_without_temporal_disambiguation() {
        assert_eq!(
            Sam3TrackerConfig::build_tracker(false),
            expected_upstream_config(false)
        );
    }
    fn tracker_build_config_matches_upstream_contract_with_temporal_disambiguation() {
        assert_eq!(
            Sam3TrackerConfig::build_tracker(true),
            expected_upstream_config(true)
        );
    }
    fn tracker_transformer_contract_matches_upstream_builder() {
        assert_eq!(
            create_tracker_transformer_config(256, 64, 72),
            expected_upstream_config(false).transformer
        );
    }

    #[test]
    fn tracker_maskmem_backbone_contract_matches_upstream_builder() {
        assert_eq!(
            create_tracker_maskmem_backbone_config(1008, 1152),
            expected_upstream_config(false).maskmem_backbone
        );
    }

    #[test]
    fn tracker_shape_spec_matches_constructed_upstream_tensor_shapes() {
        assert_eq!(
            create_shape_spec(1008, 256, 64, 14, 7),
            expected_upstream_config(false).shapes
        );
    }

    #[test]
    fn tracker_config_from_sam3_config_updates_derived_shapes_consistently() {
        let config = Sam3TrackerConfig::from_sam3_config(&tiny_config());
        assert_eq!(config.image_size, 56);
        assert_eq!(config.hidden_dim, 32);
        assert_eq!(config.memory_dim, 64);
        assert_eq!(config.backbone_stride, 14);
        assert_eq!(config.shapes.image_embedding_size, 4);
        assert_eq!(config.shapes.low_res_mask_size, 16);
        assert_eq!(config.shapes.input_mask_size, 64);
        assert_eq!(config.transformer.self_attention.feat_sizes, [4, 4]);
        assert_eq!(config.transformer.cross_attention.feat_sizes, [4, 4]);
        assert_eq!(config.prompt_encoder.image_embedding_size, [4, 4]);
        assert_eq!(config.prompt_encoder.input_image_size, [56, 56]);
        assert_eq!(config.prompt_encoder.mask_input_size, [16, 16]);
        assert_eq!(
            config.maskmem_backbone.mask_downsampler.interpol_size,
            [64, 64]
        );
        assert_eq!(
            config.shapes.obj_ptr_proj_weight_shapes,
            vec![[32, 32], [32, 32], [32, 32]]
        );
        assert_eq!(config.shapes.obj_ptr_tpos_proj_weight_shape, [64, 32]);
    }

    #[test]
    fn tracker_model_exposes_exact_builder_shapes() -> Result<()> {
        let device = candle::Device::Cpu;
        let model = Sam3TrackerModel::new(
            &Sam3TrackerConfig::build_tracker(false),
            VarBuilder::zeros(DType::F32, &device),
        )?;
        assert_eq!(model.image_embedding_size(), 72);
        assert_eq!(model.low_res_mask_size(), 288);
        assert_eq!(model.input_mask_size(), 1152);
        Ok(())
    }

    fn assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
        bundle: TrackerFixtureBundle,
        apply_temporal_disambiguation: bool,
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(bundle)?;
        let fixture = manifest.tracker_config;
        let predictor_fixture = manifest.predictor_config;
        let config = Sam3TrackerConfig::build_tracker(apply_temporal_disambiguation);
        assert_eq!(config.predictor.with_backbone, fixture.with_backbone);
        assert_eq!(config.image_size, fixture.image_size);
        assert_eq!(config.backbone_stride, fixture.backbone_stride);
        assert_eq!(config.low_res_mask_size(), fixture.low_res_mask_size);
        assert_eq!(config.shapes.input_mask_size, fixture.input_mask_size);
        assert_eq!(config.num_maskmem, fixture.num_maskmem);
        assert_eq!(
            config.max_cond_frames_in_attn,
            fixture.max_cond_frames_in_attn
        );
        assert_eq!(config.keep_first_cond_frame, fixture.keep_first_cond_frame);
        assert_eq!(
            config.memory_temporal_stride_for_eval,
            fixture.memory_temporal_stride_for_eval
        );
        assert_eq!(
            config.max_obj_ptrs_in_encoder,
            fixture.max_obj_ptrs_in_encoder
        );
        assert_eq!(
            config.non_overlap_masks_for_mem_enc,
            fixture.non_overlap_masks_for_mem_enc
        );
        assert_eq!(
            config.sigmoid_scale_for_mem_enc,
            fixture.sigmoid_scale_for_mem_enc
        );
        assert_eq!(
            config.sigmoid_bias_for_mem_enc,
            fixture.sigmoid_bias_for_mem_enc
        );
        assert_eq!(
            config.multimask_output_in_sam,
            fixture.multimask_output_in_sam
        );
        assert_eq!(
            config.multimask_output_for_tracking,
            fixture.multimask_output_for_tracking
        );
        assert_eq!(config.multimask_min_pt_num, fixture.multimask_min_pt_num);
        assert_eq!(config.multimask_max_pt_num, fixture.multimask_max_pt_num);
        assert_eq!(config.use_memory_selection, fixture.use_memory_selection);
        assert_eq!(config.mf_threshold, fixture.mf_threshold);
        assert_eq!(
            config.predictor.forward_backbone_per_frame_for_eval,
            fixture.forward_backbone_per_frame_for_eval
        );
        assert_eq!(
            config.predictor.trim_past_non_cond_mem_for_eval,
            fixture.trim_past_non_cond_mem_for_eval
        );
        assert_eq!(
            config.predictor.offload_output_to_cpu_for_eval,
            fixture.offload_output_to_cpu_for_eval
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_via_stability,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_via_stability
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_stability_delta,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_stability_delta
        );
        assert_eq!(
            config.mask_decoder.dynamic_multimask_stability_thresh,
            fixture
                .sam_mask_decoder_extra_args
                .dynamic_multimask_stability_thresh
        );
        assert_eq!(fixture.input_mask_binarize_threshold, 0.0);
        assert_eq!(fixture.video_mask_binarize_threshold, 0.5);
        assert_eq!(fixture.mask_as_output_out_scale, 20.0);
        assert_eq!(fixture.mask_as_output_out_bias, -10.0);
        assert_eq!(fixture.memory_prompt_mask_threshold, 0.0);
        assert_eq!(
            config.predictor.fill_hole_area,
            predictor_fixture.fill_hole_area
        );
        assert_eq!(
            config.predictor.clear_non_cond_mem_around_input,
            predictor_fixture.clear_non_cond_mem_around_input
        );
        assert_eq!(
            config.predictor.clear_non_cond_mem_for_multi_obj,
            predictor_fixture.clear_non_cond_mem_for_multi_obj
        );
        assert_eq!(
            config.predictor.always_start_from_first_ann_frame,
            predictor_fixture.always_start_from_first_ann_frame
        );
        assert_eq!(
            config.predictor.max_point_num_in_prompt_enc,
            predictor_fixture.max_point_num_in_prompt_enc
        );
        assert_eq!(
            config.predictor.non_overlap_masks_for_output,
            predictor_fixture.non_overlap_masks_for_output
        );
        assert_eq!(
            config.predictor.iter_use_prev_mask_pred,
            predictor_fixture.iter_use_prev_mask_pred
        );
        assert_eq!(
            config.predictor.add_all_frames_to_correct_as_cond,
            predictor_fixture.add_all_frames_to_correct_as_cond
        );
        assert_eq!(
            config.predictor.use_prev_mem_frame,
            predictor_fixture.use_prev_mem_frame
        );
        assert_eq!(
            config.predictor.use_stateless_refinement,
            predictor_fixture.use_stateless_refinement
        );
        assert_eq!(
            config
                .predictor
                .refinement_detector_cond_frame_removal_window,
            predictor_fixture.refinement_detector_cond_frame_removal_window
        );
        assert_eq!(
            config.predictor.hotstart_delay,
            predictor_fixture.hotstart_delay
        );
        assert_eq!(
            config.predictor.hotstart_unmatch_thresh,
            predictor_fixture.hotstart_unmatch_thresh
        );
        assert_eq!(
            config.predictor.hotstart_dup_thresh,
            predictor_fixture.hotstart_dup_thresh
        );
        assert_eq!(
            config.predictor.masklet_confirmation_enable,
            predictor_fixture.masklet_confirmation_enable
        );
        assert_eq!(
            config.predictor.masklet_confirmation_consecutive_det_thresh,
            predictor_fixture.masklet_confirmation_consecutive_det_thresh
        );
        assert_eq!(
            config.predictor.compile_all_components,
            predictor_fixture.compile_model
        );
        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_config_matches_default_runtime_upstream_bundle() -> Result<()> {
        assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
            TrackerFixtureBundle::Default,
            false,
        )
    }

    #[test]
    fn fixture_backed_tracker_config_matches_temporal_disambiguation_runtime_upstream_bundle(
    ) -> Result<()> {
        assert_fixture_backed_tracker_config_matches_runtime_upstream_bundle(
            TrackerFixtureBundle::TemporalDisambiguation,
            true,
        )
    }

    #[test]
    fn fixture_backed_tracker_config_matches_mem_non_overlap_runtime_upstream_bundle() -> Result<()>
    {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MemNonOverlap)?;
        let config = tracker_runtime_config_from_fixture_manifest(&manifest);
        assert!(config.non_overlap_masks_for_mem_enc);
        assert!(!config.predictor.trim_past_non_cond_mem_for_eval);
        assert!(!config.predictor.offload_output_to_cpu_for_eval);
        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_config_matches_long_history_trim_mem_runtime_upstream_bundle(
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryTrimMem)?;
        let config = tracker_runtime_config_from_fixture_manifest(&manifest);
        assert!(!config.non_overlap_masks_for_mem_enc);
        assert!(config.predictor.trim_past_non_cond_mem_for_eval);
        assert!(!config.predictor.offload_output_to_cpu_for_eval);
        Ok(())
    }

    fn assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
        bundle: TrackerFixtureBundle,
        apply_temporal_disambiguation: bool,
    ) -> Result<()> {
        let manifest = load_tracker_internal_manifest(bundle)?;
        let config = Sam3TrackerConfig::build_tracker(apply_temporal_disambiguation);

        let add_new_objects = tracker_record(&manifest, 0, "tracker_add_new_objects_input")?;
        assert_eq!(
            fixture_shape(add_new_objects, "new_object_masks_before_resize")?,
            vec![1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_dtype(add_new_objects, "new_object_masks_before_resize")?,
            "torch.bfloat16"
        );

        let frame0_track_step = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            fixture_shape(frame0_track_step, "current_vision_feats")?,
            vec![
                config.image_embedding_size() * config.image_embedding_size(),
                1,
                config.hidden_dim
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "current_vision_pos_embeds")?,
            vec![
                config.image_embedding_size() * config.image_embedding_size(),
                1,
                config.hidden_dim
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "mask_inputs")?,
            vec![
                1,
                1,
                config.shapes.input_mask_size,
                config.shapes.input_mask_size
            ]
        );
        let expected_mask_input_low_res =
            (config.shapes.input_mask_size / config.backbone_stride) * 4;
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.pred_masks")?,
            vec![
                1,
                1,
                expected_mask_input_low_res,
                expected_mask_input_low_res
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.pred_masks_high_res")?,
            vec![
                1,
                1,
                config.shapes.input_mask_size,
                config.shapes.input_mask_size
            ]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(frame0_track_step, "track_step_output.object_score_logits")?,
            vec![1, 1]
        );

        let frame0_preflight =
            tracker_record(&manifest, 0, "tracker_add_new_objects_post_preflight")?;
        assert_eq!(
            fixture_shape(frame0_preflight, "post_preflight_cond_output.pred_masks")?,
            vec![
                1,
                1,
                config.shapes.low_res_mask_size,
                config.shapes.low_res_mask_size
            ]
        );
        assert_eq!(
            fixture_shape(frame0_preflight, "post_preflight_cond_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.object_score_logits"
            )?,
            vec![1, 1]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.maskmem_features"
            )?,
            vec![
                1,
                config.memory_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );
        assert_eq!(
            fixture_shape(
                frame0_preflight,
                "post_preflight_cond_output.maskmem_pos_enc.0"
            )?,
            vec![
                1,
                config.memory_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );

        for frame_idx in 0..=3 {
            let encode_new_memory = tracker_record(&manifest, frame_idx, "encode_new_memory")?;
            assert_eq!(
                fixture_shape(encode_new_memory, "maskmem_features")?,
                vec![
                    1,
                    config.memory_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            assert_eq!(
                fixture_shape(encode_new_memory, "maskmem_pos_enc.0")?,
                vec![
                    1,
                    config.memory_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            assert_eq!(
                fixture_shape(encode_new_memory, "object_score_logits")?,
                vec![1, 1]
            );
        }

        for frame_idx in 1..=3 {
            let prep = tracker_record(&manifest, frame_idx, "prepare_memory_conditioned_features")?;
            assert_eq!(
                fixture_shape(prep, "pix_feat_with_mem")?,
                vec![
                    1,
                    config.hidden_dim,
                    config.image_embedding_size(),
                    config.image_embedding_size()
                ]
            );
            let pointer_frames = prep.metadata["selected_object_pointer_frame_indices"]
                .as_array()
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "frame {frame_idx} prepare_memory_conditioned_features missing selected_object_pointer_frame_indices"
                    ))
                })?;
            assert_eq!(
                fixture_shape(prep, "object_pointer_temporal_pos_enc")?,
                vec![pointer_frames.len(), config.memory_dim]
            );

            let track_step = tracker_record(&manifest, frame_idx, "track_step")?;
            assert_eq!(
                fixture_shape(track_step, "current_vision_feats")?,
                vec![
                    config.image_embedding_size() * config.image_embedding_size(),
                    1,
                    config.hidden_dim
                ]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.pred_masks")?,
                vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.pred_masks_high_res")?,
                vec![1, 1, config.image_size, config.image_size]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.obj_ptr")?,
                vec![1, config.hidden_dim]
            );
            assert_eq!(
                fixture_shape(track_step, "track_step_output.object_score_logits")?,
                vec![1, 1]
            );
        }

        Ok(())
    }

    #[test]
    fn fixture_backed_tracker_tensor_shapes_match_default_runtime_upstream_bundle() -> Result<()> {
        assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
            TrackerFixtureBundle::Default,
            false,
        )
    }

    #[test]
    fn fixture_backed_tracker_tensor_shapes_match_temporal_disambiguation_runtime_upstream_bundle(
    ) -> Result<()> {
        assert_fixture_backed_tracker_tensor_shapes_match_upstream_runtime_bundle(
            TrackerFixtureBundle::TemporalDisambiguation,
            true,
        )
    }

    #[test]
    fn fixture_backed_point_prompt_runtime_bundle_matches_exported_shapes() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::PointSingleClick)?;
        let config = Sam3TrackerConfig::build_tracker(false);
        let prompt_encoder = tracker_record(&manifest, 0, "sam_prompt_encoder")?;
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_inputs.points.0")?,
            vec![1, 1, 2]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_inputs.points.1")?,
            vec![1, 1]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_output.sparse_embeddings")?,
            vec![1, 2, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(prompt_encoder, "prompt_encoder_output.dense_embeddings")?,
            vec![
                1,
                config.hidden_dim,
                config.image_embedding_size(),
                config.image_embedding_size()
            ]
        );

        let mask_decoder = tracker_record(&manifest, 0, "sam_mask_decoder")?;
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.low_res_multimasks")?,
            vec![1, 3, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.ious")?,
            vec![1, 3]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.sam_output_tokens")?,
            vec![1, 3, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(mask_decoder, "mask_decoder_output.object_score_logits")?,
            vec![1, 1]
        );

        let forward_sam_heads = tracker_record(&manifest, 0, "forward_sam_heads")?;
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.low_res_masks")?,
            vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.high_res_masks")?,
            vec![1, 1, config.image_size, config.image_size]
        );
        assert_eq!(
            fixture_shape(forward_sam_heads, "forward_sam_heads_output.obj_ptr")?,
            vec![1, config.hidden_dim]
        );
        assert_eq!(
            fixture_shape(
                forward_sam_heads,
                "forward_sam_heads_output.object_score_logits"
            )?,
            vec![1, 1]
        );

        let track_step = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            fixture_shape(track_step, "track_step_output.pred_masks")?,
            vec![1, 1, config.low_res_mask_size(), config.low_res_mask_size()]
        );
        assert_eq!(
            fixture_shape(track_step, "track_step_output.pred_masks_high_res")?,
            vec![1, 1, config.image_size, config.image_size]
        );
        Ok(())
    }

    #[test]
    fn tracker_track_frame_matches_single_click_point_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_multi_click_point_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointMultiClick,
            4,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_all_points_fixture_values() -> Result<()> {
        assert_prompt_frame_point_fixture_matches(
            TrackerFixtureBundle::PointAllPoints,
            6,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_track_frame_matches_mask_prompt_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_checkpoint()? else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MaskDirect)?;
        let use_mask_stage = tracker_record(&manifest, 0, "use_mask_as_output")?;
        let track_stage = tracker_record(&manifest, 0, "track_step")?;
        assert_eq!(
            track_stage.metadata["has_mask_inputs"].as_bool(),
            Some(true)
        );
        let visual = build_fixture_visual_output(TrackerFixtureBundle::MaskDirect, use_mask_stage)?;
        let mask_input = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            track_stage.tensor_keys["mask_inputs"].as_str(),
        )?;
        let actual = model.track_frame(
            &visual,
            0,
            30,
            None,
            None,
            None,
            Some(&mask_input),
            &BTreeMap::new(),
            true,
            false,
            true,
            false,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            use_mask_stage.tensor_keys["use_mask_as_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "mask prompt low_res_masks",
            &actual.state.low_res_masks,
            &expected_low_res_masks,
            5e-4,
        )?;
        assert_tensor_close(
            "mask prompt high_res_masks",
            &actual.state.high_res_masks,
            &expected_high_res_masks,
            1e-5,
        )?;
        assert_tensor_close(
            "mask prompt iou_scores",
            &actual.state.iou_scores,
            &expected_ious,
            1e-5,
        )?;
        assert_tensor_close(
            "mask prompt obj_ptr",
            &actual.state.obj_ptr,
            &expected_obj_ptr,
            0.5,
        )?;
        assert_tensor_close(
            "mask prompt object_score_logits",
            &actual.state.object_score_logits,
            &expected_object_score_logits,
            1e-5,
        )?;
        Ok(())
    }

    #[test]
    fn tracker_mask_decoder_matches_single_click_fixture_values() -> Result<()> {
        assert_mask_decoder_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_mask_decoder_matches_multimask_disabled_sam_fixture_values() -> Result<()> {
        assert_mask_decoder_fixture_matches(
            TrackerFixtureBundle::MultimaskDisabledSam,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_single_click_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::PointSingleClick,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_all_points_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::PointAllPoints,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_forward_sam_heads_matches_multimask_disabled_sam_fixture_values() -> Result<()> {
        assert_forward_sam_heads_fixture_matches(
            TrackerFixtureBundle::MultimaskDisabledSam,
            1.0,
            1.0,
            0.2,
            0.5,
            0.5,
        )
    }

    #[test]
    fn tracker_use_mask_as_output_matches_direct_mask_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::MaskDirect)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::MaskDirect)?;
        let stage = tracker_record(&manifest, 0, "use_mask_as_output")?;
        let backbone_features = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["backbone_features"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let high_res_features = vec![
            load_tracker_fixture_tensor(
                TrackerFixtureBundle::MaskDirect,
                stage.tensor_keys["high_res_features.0"].as_str(),
            )?
            .to_dtype(DType::F32)?,
            load_tracker_fixture_tensor(
                TrackerFixtureBundle::MaskDirect,
                stage.tensor_keys["high_res_features.1"].as_str(),
            )?
            .to_dtype(DType::F32)?,
        ];
        let mask_inputs = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["mask_inputs"].as_str(),
        )?
        .to_dtype(DType::F32)?;
        let actual = model.use_mask_as_output(
            &backbone_features,
            Some(high_res_features.as_slice()),
            &mask_inputs,
            true,
        )?;
        let expected_low_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.low_res_masks"].as_str(),
        )?;
        let expected_high_res_masks = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.high_res_masks"].as_str(),
        )?;
        let expected_ious = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.ious"].as_str(),
        )?;
        let expected_obj_ptr = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.obj_ptr"].as_str(),
        )?;
        let expected_object_score_logits = load_tracker_fixture_tensor(
            TrackerFixtureBundle::MaskDirect,
            stage.tensor_keys["use_mask_as_output.object_score_logits"].as_str(),
        )?;
        assert_tensor_close(
            "use_mask_as_output low_res_masks",
            &actual.low_res_masks,
            &expected_low_res_masks,
            5e-4,
        )?;
        assert_tensor_close(
            "use_mask_as_output high_res_masks",
            &actual.high_res_masks,
            &expected_high_res_masks,
            1e-5,
        )?;
        assert_tensor_close(
            "use_mask_as_output ious",
            &actual.iou_scores,
            &expected_ious,
            1e-5,
        )?;
        assert_tensor_close(
            "use_mask_as_output obj_ptr",
            &actual.obj_ptr,
            &expected_obj_ptr,
            0.5,
        )?;
        assert_tensor_close(
            "use_mask_as_output object_score_logits",
            &actual.object_score_logits,
            &expected_object_score_logits,
            1e-5,
        )?;
        Ok(())
    }

    #[test]
    fn tracker_get_tpos_enc_matches_default_fixture_values() -> Result<()> {
        let Some(model) = load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::Default)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::Default)?;
        let stage = tracker_record(&manifest, 1, "prepare_memory_conditioned_features")?;
        let offsets = stage.metadata["selected_object_pointer_temporal_offsets"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "default fixture missing selected_object_pointer_temporal_offsets".into(),
                )
            })?
            .iter()
            .map(|value| value.as_i64().unwrap_or_default())
            .collect::<Vec<_>>();
        let max_abs_pos = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg("default fixture missing max_obj_ptrs_in_encoder".into())
            })? as usize;
        let expected = load_tracker_fixture_tensor(
            TrackerFixtureBundle::Default,
            stage.tensor_keys["object_pointer_temporal_pos_enc"].as_str(),
        )?;
        let actual = model.get_tpos_enc(
            offsets.as_slice(),
            &candle::Device::Cpu,
            Some(max_abs_pos),
            false,
        )?;
        assert_tensor_close("get_tpos_enc", &actual, &expected, 1e-2)?;
        Ok(())
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_stride1_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_stride_gt1_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryStrideGt1,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_obj_ptr_overflow_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryObjPtrOverflow,
            29,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_keep_first_cond_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryKeepFirstCond,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_prepare_memory_conditioned_features_matches_temporal_disambiguation_long_history_fixture_values(
    ) -> Result<()> {
        assert_prepare_memory_conditioned_features_fixture_matches(
            TrackerFixtureBundle::LongHistoryTemporalDisambiguation,
            28,
            1.1e-1,
        )
    }

    #[test]
    fn tracker_get_tpos_enc_matches_stride1_long_history_fixture_values() -> Result<()> {
        let Some(model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::LongHistoryStride1)?
        else {
            return Ok(());
        };
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryStride1)?;
        let stage = tracker_record(&manifest, 28, "prepare_memory_conditioned_features")?;
        let offsets =
            metadata_i64_vec(&stage.metadata, "selected_object_pointer_temporal_offsets")?;
        let max_abs_pos = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg("stride1 fixture missing max_obj_ptrs_in_encoder".into())
            })? as usize;
        let expected = load_tracker_fixture_tensor(
            TrackerFixtureBundle::LongHistoryStride1,
            stage.tensor_keys["object_pointer_temporal_pos_enc"].as_str(),
        )?;
        let actual = model.get_tpos_enc(
            offsets.as_slice(),
            &candle::Device::Cpu,
            Some(max_abs_pos),
            false,
        )?;
        assert_tensor_close("get_tpos_enc stride1", &actual, &expected, 2e-2)?;
        Ok(())
    }

    #[test]
    fn tracker_memory_conditioning_prompt_matches_stride1_long_history_fixture_values() -> Result<()>
    {
        assert_memory_conditioning_prompt_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-4,
            2e-2,
        )
    }

    #[test]
    fn tracker_memory_transformer_encoder_matches_stride1_long_history_fixture_values() -> Result<()>
    {
        assert_memory_transformer_encoder_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-1,
        )
    }

    #[test]
    fn tracker_memory_transformer_encoder_from_reconstructed_prompt_matches_stride1_long_history_fixture_values(
    ) -> Result<()> {
        assert_memory_transformer_encoder_from_reconstructed_prompt_fixture_matches(
            TrackerFixtureBundle::LongHistoryStride1,
            28,
            1e-1,
        )
    }

    #[test]
    fn tracker_object_pointer_selection_overflows_and_truncates_at_encoder_cap() -> Result<()> {
        let manifest =
            load_tracker_internal_manifest(TrackerFixtureBundle::LongHistoryObjPtrOverflow)?;
        let stage = tracker_record(&manifest, 29, "prepare_memory_conditioned_features")?;
        let max_obj_ptrs = stage.metadata["max_obj_ptrs_in_encoder"]
            .as_u64()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "object-pointer overflow fixture missing max_obj_ptrs_in_encoder".into(),
                )
            })? as usize;
        let frame_indices =
            metadata_usize_vec(&stage.metadata, "selected_object_pointer_frame_indices")?;
        let is_cond = stage.metadata["selected_object_pointer_is_conditioning"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "object-pointer overflow fixture missing selected_object_pointer_is_conditioning"
                        .into(),
                )
            })?
            .iter()
            .map(|value| value.as_bool().unwrap_or(false))
            .collect::<Vec<_>>();
        if frame_indices.len() != is_cond.len() {
            candle::bail!(
                "object-pointer overflow fixture has mismatched frame/is_cond lengths: {} vs {}",
                frame_indices.len(),
                is_cond.len()
            );
        }
        let non_cond_frames = frame_indices
            .iter()
            .zip(is_cond.iter())
            .filter_map(|(frame_idx, is_cond)| (!*is_cond).then_some(*frame_idx))
            .collect::<Vec<_>>();
        assert!(
            frame_indices.len() > max_obj_ptrs,
            "overflow fixture should exceed cap: selected {} <= cap {}",
            frame_indices.len(),
            max_obj_ptrs
        );
        assert_eq!(
            non_cond_frames.len(),
            max_obj_ptrs.saturating_sub(1),
            "overflow fixture should retain exactly cap-1 non-conditioning object pointers"
        );
        assert!(
            !non_cond_frames.contains(&13),
            "oldest non-conditioning object pointer frame should be truncated once cap is exceeded"
        );
        for expected in 14..=28 {
            assert!(
                non_cond_frames.contains(&expected),
                "expected non-conditioning object pointer frame {expected} to be retained"
            );
        }
        Ok(())
    }

    #[test]
    fn tracker_use_multimask_matches_fixture_branch_decisions() -> Result<()> {
        let Some(default_model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::PointSingleClick)?
        else {
            return Ok(());
        };
        let Some(disabled_tracking_model) = load_runtime_tracker_model_from_bundle(
            TrackerFixtureBundle::MultimaskDisabledTracking,
        )?
        else {
            return Ok(());
        };
        let Some(disabled_sam_model) =
            load_runtime_tracker_model_from_bundle(TrackerFixtureBundle::MultimaskDisabledSam)?
        else {
            return Ok(());
        };
        assert!(default_model.use_multimask(true, 1));
        assert!(!default_model.use_multimask(true, 4));
        assert!(!default_model.use_multimask(true, 6));
        assert!(disabled_tracking_model.use_multimask(true, 1));
        assert!(!disabled_tracking_model.use_multimask(false, 0));
        assert!(!disabled_sam_model.use_multimask(true, 1));
        Ok(())
    }

    #[test]
    fn default_box_bundle_routes_through_visual_prompt_before_tracker_runtime() -> Result<()> {
        let manifest = load_tracker_internal_manifest(TrackerFixtureBundle::Default)?;
        let visual_prompt_stage = tracker_record(&manifest, 0, "get_visual_prompt")?;
        let prompt_stage = tracker_record(&manifest, 0, "sam_prompt_encoder")?;
        let forward_stage = tracker_record(&manifest, 0, "forward_sam_heads")?;
        assert_eq!(
            visual_prompt_stage.metadata["input_box_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            visual_prompt_stage.metadata["created_visual_prompt"].as_bool(),
            Some(true)
        );
        assert_eq!(prompt_stage.metadata["has_boxes"].as_bool(), Some(false));
        assert_eq!(
            forward_stage.metadata["has_point_inputs"].as_bool(),
            Some(false)
        );
        Ok(())
    }
