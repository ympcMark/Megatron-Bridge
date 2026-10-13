# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline media preprocessing for native MCore MAGI-2 pretraining."""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from megatron.core.models.magi2 import Magi2Modality
from torch import Tensor


MAGI2_VIDEO_CHANNELS = 48
MAGI2_AUDIO_CHANNELS = 64
MAGI2_TEXT_CHANNELS = 5120


@dataclass(frozen=True)
class Magi2MediaRecord:
    """One raw media example described by the preprocessing JSONL manifest."""

    sample_id: int
    split: str
    video_path: Path
    audio_path: Path
    text: str


def read_magi2_media_manifest(path: str | Path) -> list[Magi2MediaRecord]:
    """Read and validate a JSONL manifest containing video, audio, and prompt fields."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"MAGI-2 media manifest does not exist: {manifest_path}")
    records: list[Magi2MediaRecord] = []
    sample_ids: set[int] = set()
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {manifest_path}") from error
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number} of {manifest_path} must contain a JSON object")
        split = item.get("split", "train")
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"line {line_number} has unsupported split {split!r}")
        sample_id = int(item.get("sample_id", len(records)))
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id} on line {line_number}")
        sample_ids.add(sample_id)
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"line {line_number} must contain non-empty text")

        paths: dict[str, Path] = {}
        for field in ("video", "audio"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"line {line_number} must contain a non-empty {field!r} path")
            media_path = Path(value)
            if not media_path.is_absolute():
                media_path = manifest_path.parent / media_path
            media_path = media_path.resolve()
            if not media_path.is_file():
                raise ValueError(f"line {line_number} {field} does not exist: {media_path}")
            paths[field] = media_path
        records.append(
            Magi2MediaRecord(
                sample_id=sample_id,
                split=split,
                video_path=paths["video"],
                audio_path=paths["audio"],
                text=text,
            )
        )
    if not records:
        raise ValueError(f"MAGI-2 media manifest contains no records: {manifest_path}")
    if not any(record.split == "train" for record in records):
        raise ValueError("MAGI-2 media manifest must contain at least one training record")
    return records


def magi2_grid_coordinates(
    shape: tuple[int, int, int],
    reference_shape: tuple[int, int, int],
    *,
    offset: tuple[int, int, int] = (0, 0, 0),
) -> Tensor:
    """Match official MAGI-2 ``get_coords`` for one modality grid."""
    if any(size <= 0 for size in (*shape, *reference_shape)):
        raise ValueError(f"MAGI-2 coordinate shapes must be positive: {shape}, {reference_shape}")
    axes = [torch.arange(size, dtype=torch.float32) + axis_offset for size, axis_offset in zip(shape, offset)]
    grids = torch.meshgrid(*axes, indexing="ij")
    positions = torch.stack(grids, dim=-1).reshape(-1, 3)
    metadata = torch.tensor((*shape, *reference_shape), dtype=torch.float32).expand(positions.shape[0], -1)
    return torch.cat((positions, metadata), dim=-1)


def magi2_latents_to_sample(
    video_latent: Tensor,
    audio_latent: Tensor,
    text_embedding: Tensor,
    *,
    sample_id: int,
    storage_dtype: torch.dtype = torch.bfloat16,
    expected_channels: dict[str, int] | None = None,
) -> dict[str, Tensor]:
    """Pack official encoder outputs into one ``magi2-latent-v1`` sample.

    ``video_latent`` is ``[C, T, H, W]``; audio and text are ``[tokens, C]``.
    The preview model uses patch size one, so video grid points become tokens
    without changing channel order.
    """
    if video_latent.ndim != 4:
        raise ValueError(f"video_latent must have shape [C, T, H, W], got {tuple(video_latent.shape)}")
    if audio_latent.ndim != 2 or text_embedding.ndim != 2:
        raise ValueError("audio_latent and text_embedding must have shape [tokens, channels]")
    video_channels, video_t, video_h, video_w = video_latent.shape
    modality_channels = {
        "video": int(video_channels),
        "audio": int(audio_latent.shape[1]),
        "text": int(text_embedding.shape[1]),
    }
    if any(size <= 0 for size in (video_t, video_h, video_w, audio_latent.shape[0], text_embedding.shape[0])):
        raise ValueError("official MAGI-2 encoder outputs must contain at least one token per modality")
    if expected_channels is not None and modality_channels != expected_channels:
        raise ValueError(
            f"official encoder channels {modality_channels} do not match expected MAGI-2 channels {expected_channels}"
        )

    video_tokens = video_latent.permute(1, 2, 3, 0).reshape(-1, video_channels)
    tensors = (video_tokens, audio_latent, text_embedding)
    input_width = max(tensor.shape[1] for tensor in tensors)
    clean_inputs = torch.cat(
        [
            F.pad(tensor.detach().to(device="cpu", dtype=storage_dtype), (0, input_width - tensor.shape[1]))
            for tensor in tensors
        ],
        dim=0,
    )

    audio_tokens = int(audio_latent.shape[0])
    text_tokens = int(text_embedding.shape[0])
    coordinates = torch.cat(
        (
            magi2_grid_coordinates((video_t, video_h, video_w), (video_t, video_h, video_w)),
            magi2_grid_coordinates((audio_tokens, 1, 1), (math.ceil(audio_tokens / 8), 1, 1)),
            magi2_grid_coordinates((text_tokens, 1, 1), (1, 1, 1), offset=(-text_tokens, 0, 0)),
        ),
        dim=0,
    )
    modality_mapping = torch.tensor(
        [int(Magi2Modality.VIDEO)] * video_tokens.shape[0]
        + [int(Magi2Modality.AUDIO)] * audio_tokens
        + [int(Magi2Modality.TEXT)] * text_tokens,
        dtype=torch.long,
    )
    return {
        "clean_inputs": clean_inputs,
        "coordinates": coordinates,
        "modality_mapping": modality_mapping,
        "sample_id": torch.tensor(sample_id, dtype=torch.long),
    }


def load_magi2_video(
    path: str | Path,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> Tensor:
    """Decode, temporally sample, center-crop, and normalize a video to ``[-1, 1]``."""
    if num_frames <= 0 or (num_frames - 1) % 8:
        raise ValueError("num_frames must be positive and have the form 8k + 1 for the official video VAE")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    decoded_frames: list[Tensor] = []
    timestamps: list[float] = []
    try:
        import av
    except ImportError:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "video decoding requires the Megatron Bridge 'diffusion' extra (PyAV or imageio-ffmpeg)"
            ) from error
        reader = imageio_ffmpeg.read_frames(str(Path(path).resolve()), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            source_fps = float(metadata["fps"])
            source_w, source_h = metadata["size"]
            for index, frame_bytes in enumerate(reader):
                frame = torch.frombuffer(bytearray(frame_bytes), dtype=torch.uint8).reshape(source_h, source_w, 3)
                decoded_frames.append(frame)
                timestamps.append(index / source_fps)
        finally:
            reader.close()
    else:
        with av.open(str(Path(path).resolve())) as container:
            if not container.streams.video:
                raise ValueError(f"media file has no video stream: {path}")
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate is not None else fps
            for index, frame in enumerate(container.decode(stream)):
                decoded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
                timestamps.append(float(frame.time) if frame.time is not None else index / source_fps)
    if not decoded_frames:
        raise ValueError(f"media file contains no decodable video frames: {path}")

    target_times = [index / fps for index in range(num_frames)]
    tolerance = 0.5 / max(float(fps), 1.0)
    if timestamps[-1] + tolerance < target_times[-1]:
        raise ValueError(
            f"video {path} is too short for {num_frames} frames at {fps:g} fps "
            f"({timestamps[-1]:.3f}s available, {target_times[-1]:.3f}s required)"
        )
    selected: list[Tensor] = []
    cursor = 0
    for target_time in target_times:
        while cursor + 1 < len(timestamps) and abs(timestamps[cursor + 1] - target_time) < abs(
            timestamps[cursor] - target_time
        ):
            cursor += 1
        selected.append(decoded_frames[cursor])

    video = torch.stack(selected).permute(0, 3, 1, 2).to(torch.float32)
    source_h, source_w = video.shape[-2:]
    scale = max(height / source_h, width / source_w)
    resized_h = max(height, math.ceil(source_h * scale))
    resized_w = max(width, math.ceil(source_w * scale))
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    video = video[:, :, top : top + height, left : left + width]
    return (video / 127.5 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def load_magi2_audio(path: str | Path, *, sample_rate: int, duration_seconds: float) -> Tensor:
    """Load audio as a padded/trimmed stereo waveform ``[1, 2, samples]``."""
    if sample_rate <= 0 or duration_seconds <= 0:
        raise ValueError("sample_rate and duration_seconds must be positive")
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as error:
        raise RuntimeError("audio preprocessing requires soundfile and scipy") from error

    waveform, source_rate = sf.read(str(Path(path).resolve()), dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError(f"media file contains no audio samples: {path}")
    if source_rate != sample_rate:
        common = math.gcd(int(source_rate), int(sample_rate))
        waveform = resample_poly(waveform, sample_rate // common, source_rate // common, axis=0)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(2, axis=1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    target_samples = int(round(duration_seconds * sample_rate))
    audio = torch.from_numpy(waveform[:target_samples]).transpose(0, 1).contiguous()
    if audio.shape[1] < target_samples:
        audio = F.pad(audio, (0, target_samples - audio.shape[1]))
    return audio.unsqueeze(0)


def validate_official_magi2_encoders(official_repo: str | Path, checkpoint_root: str | Path) -> None:
    """Fail early unless the official source tree and all three encoder checkpoints exist."""
    repository = Path(official_repo).resolve()
    checkpoint = Path(checkpoint_root).resolve()
    required = (
        repository / "inference/model/vae2_2.py",
        repository / "inference/model/qwen35.py",
        repository / "inference/pipeline/audio_decoder.py",
        checkpoint / "vae/Wan2.2_VAE.pth",
        checkpoint / "stable-audio-open-1.0/model_config.json",
        checkpoint / "stable-audio-open-1.0/model.safetensors",
        checkpoint / "text_encoder/config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    text_encoder = checkpoint / "text_encoder"
    if text_encoder.is_dir() and not any(text_encoder.glob("*.safetensors")):
        missing.append(f"{text_encoder}/*.safetensors")
    if missing:
        raise FileNotFoundError("missing official MAGI-2 encoder files:\n  " + "\n  ".join(missing))


class OfficialMagi2Encoders:
    """Thin adapter around the unmodified public MAGI-2 encoder implementations."""

    def __init__(
        self,
        official_repo: str | Path,
        checkpoint_root: str | Path,
        *,
        device: str = "cuda",
        text_max_length: int = 7000,
    ) -> None:
        validate_official_magi2_encoders(official_repo, checkpoint_root)
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("official MAGI-2 media encoding requires a CUDA device")
        repository = Path(official_repo).resolve()
        checkpoint = Path(checkpoint_root).resolve()
        torch.cuda.set_device(torch.device(device))
        repository_string = str(repository)
        if repository_string not in sys.path:
            sys.path.insert(0, repository_string)
        inference_package = importlib.import_module("inference")
        package_path = Path(inference_package.__file__).resolve()
        if repository not in package_path.parents:
            raise RuntimeError(f"an unrelated 'inference' package is already imported from {package_path}")

        vae_module = importlib.import_module("inference.model.vae2_2")
        audio_module = importlib.import_module("inference.pipeline.audio_decoder")
        text_module = importlib.import_module("inference.model.qwen35")
        self.device = torch.device(device)
        self.video_vae = vae_module.get_vae2_2(
            str(checkpoint / "vae/Wan2.2_VAE.pth"), device=device, weight_dtype=torch.float32
        )
        self.audio_vae = audio_module.SAAudioFeatureExtractor(str(checkpoint / "stable-audio-open-1.0"))
        self.text_encoder = text_module.Qwen35TextEncoder(
            str(checkpoint / "text_encoder"),
            device=device,
            precision=torch.bfloat16,
            max_length=text_max_length,
            skip_layer=2,
        )

    @property
    def audio_sample_rate(self) -> int:
        return int(self.audio_vae.sample_rate)

    @torch.inference_mode()
    def encode(self, video: Tensor, audio: Tensor, text: str) -> tuple[Tensor, Tensor, Tensor]:
        """Run the official encoders and return canonical latent layouts on CPU."""
        video_latent = self.video_vae.encode(video.to(device=self.device, dtype=torch.float32))[0]
        audio_latent = self.audio_vae.encode(audio.to(device=self.device, dtype=torch.float32))[0].transpose(0, 1)
        text_embedding = self.text_encoder.encode(text)[0]
        return video_latent.cpu(), audio_latent.cpu(), text_embedding.cpu()


def encode_magi2_media_record(
    record: Magi2MediaRecord,
    encoders: OfficialMagi2Encoders,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor]:
    """Decode one record, run official encoders, and produce a training sample."""
    video = load_magi2_video(record.video_path, num_frames=num_frames, fps=fps, height=height, width=width)
    audio = load_magi2_audio(
        record.audio_path,
        sample_rate=encoders.audio_sample_rate,
        duration_seconds=num_frames / fps,
    )
    video_latent, audio_latent, text_embedding = encoders.encode(video, audio, record.text)
    return magi2_latents_to_sample(
        video_latent,
        audio_latent,
        text_embedding,
        sample_id=record.sample_id,
        storage_dtype=storage_dtype,
        expected_channels={
            "video": MAGI2_VIDEO_CHANNELS,
            "audio": MAGI2_AUDIO_CHANNELS,
            "text": MAGI2_TEXT_CHANNELS,
        },
    )


__all__ = [
    "MAGI2_AUDIO_CHANNELS",
    "MAGI2_TEXT_CHANNELS",
    "MAGI2_VIDEO_CHANNELS",
    "Magi2MediaRecord",
    "OfficialMagi2Encoders",
    "encode_magi2_media_record",
    "load_magi2_audio",
    "load_magi2_video",
    "magi2_grid_coordinates",
    "magi2_latents_to_sample",
    "read_magi2_media_manifest",
    "validate_official_magi2_encoders",
]
