"""
DCASE 2025 fine-tuned baseline model — Modular API for ASC inference.
Using the same spectrogram generation method as training (AugmentMelSTFT).
"""

from typing import Optional, List
import torch
import torchaudio
import importlib.resources as pkg_resources
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Model and resource imports
from Jeong_SEOULTECH_task1.models.net import get_model
from Jeong_SEOULTECH_task1.models.multi_device_model import MultiDeviceModelContainer
from Jeong_SEOULTECH_task1.models.processor import AugmentMelSTFT
from Jeong_SEOULTECH_task1 import ckpts


class Config:
    """Configuration for audio preprocessing and model structure."""

    # Audio parameters (match train_kd.py)
    orig_sample_rate = 44100  # Original file sample rate
    sample_rate = 32000       # Target sample rate (handled by AugmentMelSTFT)

    # Spectrogram parameters (aligned with training)
    n_mels = 256
    win_length = 2048
    hopsize = 744
    n_fft = 2048
    fmin = 0.0
    fmax = None

    # Model architecture
    n_classes = 10
    in_channels = 1
    base_channels = 32
    channels_multiplier = 1.8
    expansion_rate = 2.1

    # Device IDs
    device_ids = ['a', 'b', 'c', 's1', 's2', 's3']


class Baseline(torch.nn.Module):
    """
    DCASE 2025 Task 1 Baseline inference class for ASC with multi-device support.
    Uses AugmentMelSTFT preprocessing aligned with training code.
    """

    def __init__(self, config: Config, use_multi_device: bool = True):
        super().__init__()
        self.config = config
        self.use_multi_device = use_multi_device

        # Preprocessing: mel spectrogram transform (aligned with training)
        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
            sr=config.sample_rate,
            win_length=config.win_length,
            hopsize=config.hopsize,
            n_fft=config.n_fft,
            freqm=0,  # No frequency masking during inference
            timem=0,  # No time masking during inference
            htk=False,
            fmin=config.fmin,
            fmax=config.fmax,
            norm=1,
            fmin_aug_range=1,  # No augmentation during inference
            fmax_aug_range=1   # No augmentation during inference
        )

        # Backbone model
        base_model = get_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate
        )

        if use_multi_device:
            # Multi-device wrapper: allows device-specific models
            self.model = MultiDeviceModelContainer(base_model, devices=config.device_ids)
        else:
            # Use single base model for all devices
            self.model = base_model
            
        self.model.eval()
        self.model.half()  # use float16 to meet complexity constraints

        self.class_order = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]

    def mel_forward(self, x: Tensor) -> Tensor:
        """
        Convert raw waveform to log-mel spectrogram using AugmentMelSTFT.
        
        Args:
            x: Tensor of shape [B, 1, n_samples] or [B, n_samples]
        Returns:
            Tensor of shape [B, n_mels, T]
        """
        # Remove channel dimension if present for AugmentMelSTFT
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [B, 1, n_samples] -> [B, n_samples]
        
        # Generate mel spectrogram
        mel = self.mel(x)  # [B, n_mels, T]
        return mel.half()

    def preprocess(self, waveform: Tensor) -> Tensor:
        """
        Convert raw waveform to log-mel spectrogram.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
        Returns:
            Tensor of shape [B, 1, n_mels, T]
        """
        mel = self.mel_forward(waveform)  # [B, n_mels, T]
        return mel.unsqueeze(1)  # [B, 1, n_mels, T]

    def forward(self, waveform: Tensor, device_ids: Optional[List[str]] = None) -> Tensor:
        """
        Perform forward pass for a batch of waveforms.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
            device_ids: List of device identifiers (length B) - only used for multi-device models

        Returns:
            Tensor of shape [B, n_classes]
        """
        with torch.no_grad():
            mel = self.preprocess(waveform)  # [B, 1, n_mels, T]
            if self.use_multi_device and device_ids is not None:
                logits = self.model(mel, device_ids)  # [B, n_classes]
            else:
                logits = self.model(mel)  # [B, n_classes]
        return logits


def load_model(model_file_path: Optional[str] = None, use_multi_device: bool = True) -> Baseline:
    """
    Load the baseline model from a checkpoint.

    Args:
        model_file_path: Optional path to a .ckpt file. If None, uses the default packaged checkpoint.
        use_multi_device: Whether to use multi-device model container or single base model.

    Returns:
        A Baseline model instance with loaded weights.
    """
    config = Config()
    model = Baseline(config, use_multi_device=use_multi_device)

    # Use default checkpoint from package resources if no path is given
    if model_file_path is None:
        with pkg_resources.path(ckpts, "acc=0.4664.ckpt") as ckpt_path:
            model_file_path = str(ckpt_path)

    # Load checkpoint to CPU (compatible with CPU inference)
    ckpt = torch.load(model_file_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in ckpt:
        # PyTorch Lightning checkpoint format
        state_dict = ckpt["state_dict"]
        
        if use_multi_device:
            # Extract multi-device model weights
            if any(k.startswith("multi_device_model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("multi_device_model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("multi_device_model.")
                }
            else:
                raise ValueError("Multi-device model weights not found in checkpoint. Use use_multi_device=False for base model checkpoints.")
        else:
            # Extract base model weights
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
            elif any(k.startswith("multi_device_model.base_model.") for k in state_dict.keys()):
                # Extract base model from multi-device checkpoint
                state_dict = {
                    k.replace("multi_device_model.base_model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("multi_device_model.base_model.")
                }
            else:
                # Assume direct model state dict
                pass
    else:
        # Direct model state dict
        state_dict = ckpt

    model.model.load_state_dict(state_dict, strict=True)
    model.model.half()
    return model


def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: Baseline,
    num_workers: int = 16,
    batch_size: int = 256
) -> List[Tensor]:
    """
    Load and preprocess audio files in parallel with batch-wise mel spectrogram computation.
    Uses original sample rate (44100Hz) as AugmentMelSTFT handles sample rate internally.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    assert len(file_paths) == len(device_ids)

    device = next(model.parameters()).device

    def _load(indexed_path):
        path, idx = indexed_path
        waveform, sr = torchaudio.load(path)              # [channels, samples]
        waveform = waveform.mean(dim=0)                   # mono: [samples]
        return idx, waveform, sr

    # Step 1: Load & mono-mix in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        loaded = list(tqdm(
            exe.map(_load, zip(file_paths, range(len(file_paths)))),
            total=len(file_paths),
            desc="Loading files"
        ))
    # loaded = list of (original_idx, waveform, sample_rate)

    # Step 2: Group by original sample rate for batch processing
    sr_groups = defaultdict(list)
    for idx, waveform, sr in loaded:
        sr_groups[sr].append((idx, waveform))

    inputs: List[Tensor] = [None] * len(file_paths)  # final output buffer

    # Step 3: Batch process each group (no resampling - AugmentMelSTFT handles sample rate)
    print("Batched Mel Processing (original sample rate)...")
    for sr, items in sr_groups.items():
        print(f"Processing SR={sr} with {len(items)} files")
        for i in tqdm(range(0, len(items), batch_size), desc=f"SR={sr}", leave=False):
            chunk = items[i:i + batch_size]
            indices, waveforms = zip(*chunk)

            # Pad waveforms to the same length → [max_len, B]
            padded = pad_sequence(waveforms, batch_first=False)

            # Reshape → [B, samples]
            batch_wave = padded.transpose(0, 1).to(device)

            # Mel computation for batch using model's mel_forward (no resampling)
            with torch.no_grad():
                mel_batch = model.mel_forward(batch_wave)  # [B, n_mels, T]

            # Store each preprocessed mel in original order
            for mel, idx in zip(mel_batch, indices):
                # Add batch and channel dimensions: [n_mels, T] -> [1, 1, n_mels, T]
                inputs[idx] = mel.unsqueeze(0).unsqueeze(0).cpu()

    return inputs


def get_model_for_device(
    model: Baseline,
    device_id: str
) -> torch.nn.Module:
    """
    Extract the device model corresponding to a specific device ID.

    Args:
        model: Baseline model instance.
        device_id: Device identifier string (e.g., 's1').

    Returns:
        The device model (nn.Module) associated with the given device.
    """
    if model.use_multi_device:
        device_model = model.model.get_model_for_device(device_id)
        device_model.half()  # ensure float16 to meet complexity constraints
        return device_model
    else:
        # For single base model, return the same model for all devices
        model.model.half()
        return model.model


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64,
    use_multi_device: bool = False  # Default to False for base model
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using device-specific models.

    Files are grouped by device ID and processed in batches.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.ckpt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.
        use_multi_device: Whether to use multi-device model container.

    Returns:
        A tuple (logits, class_order)
        - logits: List of tensors, one per file, each of shape [n_classes]
        - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path, use_multi_device=use_multi_device).to(device)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

    if not use_multi_device:
        # For single base model, process all inputs together
        outputs = [None] * len(inputs)
        
        # Process in batches
        for i in tqdm(range(0, len(inputs), batch_size), desc="Batched inference"):
            chunk_inputs = inputs[i:i + batch_size]
            chunk_indices = list(range(i, min(i + batch_size, len(inputs))))
            
            # Stack inputs for batch processing
            mels = [inp.squeeze(0).squeeze(0) for inp in chunk_inputs]  # Remove batch dims: [n_mels, T]
            
            # Pad to same length → [max_T, batch, n_mels]
            padded = pad_sequence([m.T for m in mels], batch_first=False)
            
            # Reshape → [batch, 1, n_mels, max_T]
            batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)
            
            with torch.no_grad():
                logits = model.model(batch).cpu()  # [B, n_classes]
            
            # Store outputs
            for logit, idx in zip(logits, chunk_indices):
                outputs[idx] = logit
                
        return outputs, model.class_order
    
    else:
        # Step 2: Group by device ID, squeeze each mel to [n_mels, T]
        groups = defaultdict(list)
        for idx, (mel, dev) in enumerate(zip(inputs, device_ids)):
            mel_squeezed = mel.squeeze(0).squeeze(0)  # → [n_mels, T]
            groups[dev].append((mel_squeezed, idx))

        outputs = [None] * len(inputs)  # Placeholder for final predictions

        # Step 3: For each device, batch and infer
        for dev, items in tqdm(groups.items(), desc="Batched inference"):
            submodel = get_model_for_device(model, dev)

            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                mels, indices = zip(*chunk)  # List of [n_mels, T_i]

                # Pad to same length → [max_T, batch, n_mels]
                # This is to potentially also support files of varying length. However,
                # all files in the TAU dataset are exactly one second in length.
                padded = pad_sequence([m.T for m in mels], batch_first=False)

                # Reshape → [batch, 1, n_mels, max_T]
                batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)

                with torch.no_grad():
                    logits = submodel(batch).cpu()  # [B, n_classes]

                # Scatter outputs back in original file order
                for logit, idx in zip(logits, indices):
                    outputs[idx] = logit

        return outputs, model.class_order 