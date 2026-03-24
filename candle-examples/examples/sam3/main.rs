#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_transformers::models::sam3;
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

#[derive(Parser, Debug)]
struct Args {
    /// Optional path to the upstream `sam3.pt` checkpoint.
    #[arg(long)]
    checkpoint: Option<String>,

    /// Optional path to a `tokenizer.json` compatible with the upstream text encoder.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Optional image path. Present for the eventual image-grounding API.
    #[arg(long)]
    image: Option<String>,

    /// Optional text prompt. Present for the eventual image-grounding API.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    print_config: bool,
}

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

fn get_tokenizer(tokenizer: &str, context_length: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
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

fn run_text_encoder(
    model: &sam3::Sam3ImageModel,
    prompt: &str,
    tokenizer_path: &str,
    context_length: usize,
    device: &Device,
) -> Result<()> {
    let tokenizer = get_tokenizer(tokenizer_path, context_length)?;
    let (input_ids, attention_mask) = tokenize_prompt(prompt, &tokenizer, device)?;
    let encoding = model.encode_text_tokens(&input_ids, &attention_mask)?;
    println!("tokenized prompt:");
    println!("  text: {prompt}");
    println!("  input_ids: {:?}", input_ids.to_vec2::<u32>()?);
    println!("  attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);
    println!("text encoder output:");
    println!("  padding mask shape: {:?}", encoding.attention_mask.dims());
    println!(
        "  input embeddings shape: {:?}",
        encoding.input_embeddings.dims()
    );
    println!("  resized memory shape: {:?}", encoding.memory.dims());
    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let config = sam3::Config::default();
    let checkpoint_source = args
        .checkpoint
        .as_ref()
        .map(|path| sam3::Sam3CheckpointSource::upstream_pth(path));

    println!("sam3 scaffold example");
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

    let model = if let Some(checkpoint) = checkpoint_source.as_ref() {
        let model =
            sam3::Sam3ImageModel::from_checkpoint_source(&config, checkpoint, DType::F32, &device)?;
        println!(
            "checkpoint opened with state key `{}` and image-model namespace remap",
            sam3::UPSTREAM_SAM3_STATE_KEY
        );
        Some(model)
    } else {
        None
    };

    if let Some(image_path) = args.image.as_ref() {
        let Some(model) = model.as_ref() else {
            anyhow::bail!(
                "loading an image into typed SAM3 state currently requires `--checkpoint <sam3.pt>`"
            );
        };
        let (image, _h, _w) =
            candle_examples::load_image(image_path, Some(config.image.image_size))?;
        let image = image.to_device(&device)?;
        let mut state = model.set_image(&image)?;
        if let Some(prompt) = args.prompt.as_ref() {
            state = state.with_text_prompt(prompt.clone());
        }
        println!("prepared typed image state:\n{state:#?}");
    }

    if let Some(prompt) = args.prompt.as_ref() {
        let tokenizer = args.tokenizer.as_deref().ok_or_else(|| {
            E::msg("encoding a SAM3 text prompt requires `--tokenizer <tokenizer.json>`")
        })?;
        let Some(model) = model.as_ref() else {
            anyhow::bail!(
                "running the SAM3 text encoder currently requires `--checkpoint <sam3.pt>`"
            );
        };
        run_text_encoder(
            model,
            prompt,
            tokenizer,
            config.text.context_length,
            &device,
        )?;
        anyhow::bail!(
            "the sam3 example now supports text tokenization and text-encoder execution, but image grounding and rendering are not implemented yet"
        );
    }

    Ok(())
}
