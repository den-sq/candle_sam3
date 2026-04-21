use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle::{DType, Device, Result, Tensor};
use image::ImageReader;

use super::super::{image::ImageSize, Config};

#[derive(Debug, Clone)]
pub enum VideoSource {
    TensorFrames(Vec<Tensor>),
    ImageFolder(PathBuf),
    ImageFile(PathBuf),
    VideoFile(PathBuf),
}

impl VideoSource {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Ok(Self::ImageFolder(path.to_path_buf()));
        }
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());
        match ext.as_deref() {
            Some("jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp") => {
                Ok(Self::ImageFile(path.to_path_buf()))
            }
            Some("mp4" | "avi" | "mov" | "mkv" | "webm") => Ok(Self::VideoFile(path.to_path_buf())),
            _ => candle::bail!("unsupported video source path {}", path.display()),
        }
    }

    pub(crate) fn into_frame_source(self, config: &Config) -> Result<Box<dyn FrameSource>> {
        match self {
            Self::TensorFrames(frames) => Ok(Box::new(TensorFrameSource::new(frames)?)),
            Self::ImageFolder(path) => Ok(Box::new(ImageFolderFrameSource::new(
                sorted_image_paths(&path)?,
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
            Self::ImageFile(path) => Ok(Box::new(ImageFolderFrameSource::new(
                vec![path],
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
            Self::VideoFile(path) => Ok(Box::new(VideoFileFrameSource::new(
                path,
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
        }
    }
}

pub trait FrameSource {
    fn frame_count(&self) -> usize;
    fn video_size(&self) -> ImageSize;
    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor>;
    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()>;
    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>);
    fn loaded_frame_count(&self) -> usize;
    fn close(&mut self);
}

#[derive(Debug, Clone)]
pub(crate) struct FrameBlob {
    pub(crate) data: Vec<f32>,
    pub(crate) frame_size: ImageSize,
}

impl FrameBlob {
    fn to_tensor(&self, target_device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            self.data.clone(),
            (3, self.frame_size.height, self.frame_size.width),
            &Device::Cpu,
        )?
        .to_device(target_device)
    }
}

struct TensorFrameSource {
    frames: Vec<Tensor>,
    video_size: ImageSize,
}

impl TensorFrameSource {
    fn new(frames: Vec<Tensor>) -> Result<Self> {
        if frames.is_empty() {
            candle::bail!("tensor frame source requires at least one frame")
        }
        let (channels, height, width) = match frames[0].rank() {
            3 => frames[0].dims3()?,
            4 => {
                let (_batch, channels, height, width) = frames[0].dims4()?;
                (channels, height, width)
            }
            rank => candle::bail!("expected CHW or BCHW frame tensor, got rank {}", rank),
        };
        if channels != 3 {
            candle::bail!(
                "tensor frame source expects RGB frames, got {} channels",
                channels
            )
        }
        Ok(Self {
            frames,
            video_size: ImageSize::new(height, width),
        })
    }
}

impl FrameSource for TensorFrameSource {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.frames
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} out of bounds", frame_idx)))?
            .to_device(target_device)
    }

    fn prefetch(&mut self, _frame_indices: &[usize]) -> Result<()> {
        Ok(())
    }

    fn evict_except(&mut self, _keep_frame_indices: &BTreeSet<usize>) {}

    fn loaded_frame_count(&self) -> usize {
        self.frames.len()
    }

    fn close(&mut self) {}
}

struct ImageFolderFrameSource {
    image_paths: Vec<PathBuf>,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    cache: HashMap<usize, FrameBlob>,
    video_size: ImageSize,
}

impl ImageFolderFrameSource {
    fn new(
        image_paths: Vec<PathBuf>,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        if image_paths.is_empty() {
            candle::bail!("image frame source requires at least one image path")
        }
        let first = image_paths[0].clone();
        let image = ImageReader::open(&first)?
            .decode()
            .map_err(candle::Error::wrap)?
            .to_rgb8();
        let (width, height) = image.dimensions();
        Ok(Self {
            image_paths,
            image_size,
            image_mean,
            image_std,
            cache: HashMap::new(),
            video_size: ImageSize::new(height as usize, width as usize),
        })
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        let path = self
            .image_paths
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} out of bounds", frame_idx)))?;
        let blob = load_frame_blob(
            path,
            self.image_size,
            self.image_mean,
            self.image_std,
            self.video_size,
        )?;
        self.cache.insert(frame_idx, blob);
        Ok(())
    }
}

impl FrameSource for ImageFolderFrameSource {
    fn frame_count(&self) -> usize {
        self.image_paths.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} not cached", frame_idx)))?
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

struct VideoFileFrameSource {
    video_path: PathBuf,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    cache: HashMap<usize, FrameBlob>,
    video_size: ImageSize,
    frame_count: usize,
}

impl VideoFileFrameSource {
    fn new(
        video_path: PathBuf,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        let metadata = probe_video_file(&video_path)?;
        Ok(Self {
            video_path,
            image_size,
            image_mean,
            image_std,
            cache: HashMap::new(),
            video_size: metadata.video_size,
            frame_count: metadata.frame_count,
        })
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        if frame_idx >= self.frame_count {
            candle::bail!(
                "frame_idx {} out of bounds for video with {} frames",
                frame_idx,
                self.frame_count
            );
        }
        let blob = decode_video_frame_blob(
            &self.video_path,
            frame_idx,
            self.image_size,
            self.image_mean,
            self.image_std,
            self.video_size,
        )?;
        self.cache.insert(frame_idx, blob);
        Ok(())
    }
}

impl FrameSource for VideoFileFrameSource {
    fn frame_count(&self) -> usize {
        self.frame_count
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} not cached", frame_idx)))?
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

fn load_frame_blob(
    image_path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    if matches!(
        image_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg")
    ) {
        if let Ok(blob) = load_jpeg_frame_blob_via_pillow(
            image_path,
            image_size,
            image_mean,
            image_std,
            expected_video_size,
        ) {
            return Ok(blob);
        }
    }
    let image = ImageReader::open(image_path)?
        .decode()
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &image_path.display().to_string(),
    )
}

fn load_jpeg_frame_blob_via_pillow(
    image_path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let python = find_pillow_python().ok_or_else(|| {
        candle::Error::Msg("no Pillow-capable python interpreter found".to_owned())
    })?;
    let script = r#"
import struct
import sys
from PIL import Image

image_path = sys.argv[1]
image_size = int(sys.argv[2])

image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
if image.size != (image_size, image_size):
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
raw = image.tobytes()
sys.stdout.buffer.write(struct.pack("<II", orig_w, orig_h))
sys.stdout.buffer.write(raw)
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(image_path)
        .arg(image_size.to_string())
        .output()
        .map_err(candle::Error::wrap)?;
    if !output.status.success() {
        candle::bail!(
            "Pillow frame load failed for {} via {}: {}",
            image_path.display(),
            python.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.len() < 8 {
        candle::bail!(
            "Pillow frame load returned truncated output for {}",
            image_path.display()
        );
    }
    let width = u32::from_le_bytes(output.stdout[0..4].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(output.stdout[4..8].try_into().unwrap()) as usize;
    let current_size = ImageSize::new(height, width);
    if current_size != expected_video_size {
        candle::bail!(
            "frame {} has size {}x{} but the session expects {}x{}",
            image_path.display(),
            current_size.height,
            current_size.width,
            expected_video_size.height,
            expected_video_size.width
        );
    }
    let expected_bytes = image_size * image_size * 3;
    let raw = &output.stdout[8..];
    if raw.len() != expected_bytes {
        candle::bail!(
            "Pillow frame load returned {} bytes for resized frame {}, expected {}",
            raw.len(),
            image_path.display(),
            expected_bytes
        );
    }
    let image = Tensor::from_vec(raw.to_vec(), (image_size, image_size, 3), &Device::Cpu)?
        .permute((2, 0, 1))?;
    let normalized = normalize_image_for_sam3(
        &(image.to_dtype(DType::F32)?.unsqueeze(0)? / 255.)?,
        image_mean,
        image_std,
    )?
    .squeeze(0)?;
    Ok(FrameBlob {
        data: normalized.flatten_all()?.to_vec1::<f32>()?,
        frame_size: ImageSize::square(image_size),
    })
}

fn find_pillow_python() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("SAM3_PILLOW_PYTHON").map(PathBuf::from) {
        candidates.push(path);
    }
    candidates.push(PathBuf::from(".venv/bin/python"));
    candidates.push(PathBuf::from(
        "/home/dnorthover/ChengCode/candle_sam3/.venv/bin/python",
    ));
    candidates.push(PathBuf::from("python3"));
    candidates.into_iter().find(|candidate| {
        candidate.is_absolute() || candidate.exists() || candidate == Path::new("python3")
    })
}

fn frame_blob_from_rgb_image(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
) -> Result<FrameBlob> {
    frame_blob_from_rgb_image_with_filter(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        source_label,
        image::imageops::FilterType::CatmullRom,
    )
}

pub(crate) fn frame_blob_from_rgb_image_with_filter(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
    resize_filter: image::imageops::FilterType,
) -> Result<FrameBlob> {
    let (width, height) = image.dimensions();
    let current_size = ImageSize::new(height as usize, width as usize);
    if current_size != expected_video_size {
        candle::bail!(
            "frame {} has size {}x{} but the session expects {}x{}",
            source_label,
            current_size.height,
            current_size.width,
            expected_video_size.height,
            expected_video_size.width
        );
    }

    let resized =
        if expected_video_size.height == image_size && expected_video_size.width == image_size {
            image
        } else {
            image::imageops::resize(&image, image_size as u32, image_size as u32, resize_filter)
        };
    let image = Tensor::from_vec(
        resized.into_raw(),
        (image_size, image_size, 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?;
    let normalized = normalize_image_for_sam3(
        &(image.to_dtype(DType::F32)?.unsqueeze(0)? / 255.)?,
        image_mean,
        image_std,
    )?
    .squeeze(0)?;
    Ok(FrameBlob {
        data: normalized.flatten_all()?.to_vec1::<f32>()?,
        frame_size: ImageSize::square(image_size),
    })
}

#[derive(Debug)]
struct VideoProbeMetadata {
    video_size: ImageSize,
    frame_count: usize,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeStream {
    width: Option<usize>,
    height: Option<usize>,
    nb_frames: Option<String>,
    nb_read_frames: Option<String>,
    duration: Option<String>,
    r_frame_rate: Option<String>,
}

fn probe_video_file(video_path: &Path) -> Result<VideoProbeMetadata> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,duration,r_frame_rate",
            "-of",
            "json",
        ])
        .arg(video_path)
        .output()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to run ffprobe for {}: {}",
                video_path.display(),
                err
            ))
        })?;
    if !output.status.success() {
        candle::bail!(
            "ffprobe failed for {}: {}",
            video_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let parsed: FfprobeOutput = serde_json::from_slice(&output.stdout).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to parse ffprobe output for {}: {}",
            video_path.display(),
            err
        ))
    })?;
    let stream = parsed.streams.into_iter().next().ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe found no video stream in {}",
            video_path.display()
        ))
    })?;
    let width = stream.width.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report width for {}",
            video_path.display()
        ))
    })?;
    let height = stream.height.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report height for {}",
            video_path.display()
        ))
    })?;
    let frame_count = parse_frame_count(&stream).ok_or_else(|| {
        candle::Error::Msg(format!(
            "could not determine frame count for {} from ffprobe metadata",
            video_path.display()
        ))
    })?;
    if frame_count == 0 {
        candle::bail!(
            "video {} contains zero readable frames",
            video_path.display()
        );
    }
    Ok(VideoProbeMetadata {
        video_size: ImageSize::new(height, width),
        frame_count,
    })
}

fn parse_frame_count(stream: &FfprobeStream) -> Option<usize> {
    parse_optional_usize(stream.nb_read_frames.as_deref())
        .or_else(|| parse_optional_usize(stream.nb_frames.as_deref()))
        .or_else(|| {
            let duration = stream.duration.as_deref()?.parse::<f64>().ok()?;
            let fps = parse_rational_f64(stream.r_frame_rate.as_deref()?)?;
            let approx = (duration * fps).round();
            (approx.is_finite() && approx > 0.0).then_some(approx as usize)
        })
}

fn parse_optional_usize(value: Option<&str>) -> Option<usize> {
    let value = value?;
    if value == "N/A" {
        return None;
    }
    value.parse::<usize>().ok().filter(|value| *value > 0)
}

fn parse_rational_f64(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    (denominator != 0.0).then_some(numerator / denominator)
}

fn decode_video_frame_blob(
    video_path: &Path,
    frame_idx: usize,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let select_filter = format!("select=eq(n\\,{frame_idx})");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(video_path)
        .args([
            "-vf",
            &select_filter,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to run ffmpeg for {} frame {}: {}",
                video_path.display(),
                frame_idx,
                err
            ))
        })?;
    if !output.status.success() {
        candle::bail!(
            "ffmpeg failed for {} frame {}: {}",
            video_path.display(),
            frame_idx,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.is_empty() {
        candle::bail!(
            "ffmpeg produced no bytes for {} frame {}",
            video_path.display(),
            frame_idx
        );
    }
    let image = image::load(Cursor::new(output.stdout), image::ImageFormat::Png)
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &format!("{}#{}", video_path.display(), frame_idx),
    )
}

fn sorted_image_paths(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut image_paths = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp"
                    )
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if image_paths.is_empty() {
        candle::bail!("no image files found in {}", dir_path.display())
    }

    if image_paths.iter().all(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<usize>().ok())
            .is_some()
    }) {
        image_paths.sort_by_key(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
    } else {
        image_paths.sort_by(|lhs, rhs| lhs.file_name().cmp(&rhs.file_name()));
    }

    Ok(image_paths)
}

fn normalize_image_for_sam3(
    image_bchw: &Tensor,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<Tensor> {
    let device = image_bchw.device();
    let mean = Tensor::from_vec(image_mean.to_vec(), (1, 3, 1, 1), device)?;
    let std = Tensor::from_vec(image_std.to_vec(), (1, 3, 1, 1), device)?;
    image_bchw.broadcast_sub(&mean)?.broadcast_div(&std)
}
