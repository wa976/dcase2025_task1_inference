"""
DCASE 2025 Student Device-Specific fine-tuned model — Modular API for ASC inference.
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
import copy

# Model and resource imports
from Jeong_SEOULTECH_task1.models.net import get_model
from Jeong_SEOULTECH_task1.models.processor import AugmentMelSTFT
from Jeong_SEOULTECH_task1 import ckpts


class MultiDeviceStudentContainer(torch.nn.Module):
    """
    Student 모델을 여러 디바이스별로 관리하는 컨테이너
    각 디바이스마다 별도의 Student 모델을 유지하고, unseen device는 원본 모델 사용
    """
    def __init__(self, base_student_model, device_ids):
        super().__init__()
        self.device_ids = device_ids
        
        # 원본 pre-trained 모델을 unseen device용으로 보관
        self.original_model = copy.deepcopy(base_student_model)
        
        # 각 디바이스별로 별도의 Student 모델 생성
        self.device_models = torch.nn.ModuleDict()
        
        for device_id in device_ids:
            # Base model을 복사해서 각 디바이스별 모델 생성
            device_model = copy.deepcopy(base_student_model)
            self.device_models[device_id] = device_model
            
        print(f"✅ Created {len(device_ids)} device-specific Student models: {device_ids}")
        print(f"✅ Stored original pre-trained model for unseen devices")
    
    def get_model_for_device(self, device_id):
        """특정 디바이스의 모델 반환"""
        if device_id not in self.device_models:
            # Unseen device의 경우 원본 pre-trained 모델 반환
            print(f"⚠️  Using original pre-trained model for unseen device: {device_id}")
            return self.original_model
        return self.device_models[device_id]
    
    def get_original_model(self):
        """원본 pre-trained 모델 반환 (unseen device용)"""
        return self.original_model
    
    def forward(self, x, devices=None):
        """
        Forward pass - 배치의 모든 샘플이 같은 디바이스 또는 mixed device batch 처리
        """
        if devices is None:
            raise ValueError("Devices must be specified for device-specific forward pass")
        
        batch_size = x.size(0)
        
        # 배치 내 모든 샘플이 같은 디바이스에서 온 경우
        unique_devices = list(set(devices))
        if len(unique_devices) == 1:
            device_id = unique_devices[0]
            model = self.get_model_for_device(device_id)
            return model(x)
        
        # Mixed device batch 처리
        else:
            # 각 디바이스별로 샘플들을 그룹화
            device_groups = {}
            for i, device in enumerate(devices):
                if device not in device_groups:
                    device_groups[device] = []
                device_groups[device].append(i)
            
            # 결과를 저장할 텐서들 초기화
            all_logits = []
            original_indices = []
            
            # 각 디바이스 그룹별로 forward pass 수행
            for device_id, indices in device_groups.items():
                # 해당 디바이스의 샘플들만 추출
                device_x = x[indices]
                
                # 해당 디바이스의 모델로 예측
                model = self.get_model_for_device(device_id)
                device_logits = model(device_x)
                
                all_logits.append(device_logits)
                original_indices.extend(indices)
            
            # 결과들을 원래 순서대로 재정렬
            combined_logits = torch.cat(all_logits, dim=0)
            
            # 원래 순서로 복원
            reorder_indices = torch.tensor(original_indices, device=x.device)
            sorted_indices = torch.argsort(reorder_indices)
            
            final_logits = combined_logits[sorted_indices]
            
            return final_logits


class Config:
    """Configuration for audio preprocessing and model structure."""

    # Audio parameters
    sample_rate = 32000

    # Spectrogram parameters (matching AugmentMelSTFT training setup)
    win_length = 2048
    hopsize = 744
    n_fft = 2048
    n_mels = 256
    fmin = 0.0
    fmax = None
    norm = 1
    fmin_aug_range = 1
    fmax_aug_range = 1

    # Model architecture (Student model parameters)
    n_classes = 10
    in_channels = 1
    base_channels = 32
    channels_multiplier = 1.8
    expansion_rate = 2.1

    # Device IDs (trained devices)
    train_device_ids = ['a', 'b', 'c', 's1', 's2', 's3']
    # All device IDs (including unseen)
    all_device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']


class Baseline(torch.nn.Module):
    """
    DCASE 2025 Task 1 Student Device-Specific inference class for ASC.
    Includes log-mel preprocessing and multi-device student model container.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Preprocessing: mel spectrogram transform (matching training setup)
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
            norm=config.norm,
            fmin_aug_range=config.fmin_aug_range,
            fmax_aug_range=config.fmax_aug_range
        )

        # Student backbone model
        base_student_model = get_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channels_multiplier=config.channels_multiplier,
            expansion_rate=config.expansion_rate
        )

        # Multi-device student wrapper
        self.model = MultiDeviceStudentContainer(
            base_student_model, 
            device_ids=config.train_device_ids
        )
        self.model.eval()
        self.model.half()  # use float16 to meet complexity constraints

        self.class_order = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]

    def preprocess(self, waveform: Tensor) -> Tensor:
        """
        Convert raw waveform to log-mel spectrogram using AugmentMelSTFT.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
        Returns:
            Tensor of shape [B, 1, n_mels, T]
        """
        # Remove channel dimension for AugmentMelSTFT
        waveform = waveform.squeeze(1)  # [B, 1, n_samples] -> [B, n_samples]
        
        # AugmentMelSTFT returns log mel spectrogram directly
        log_mel = self.mel(waveform)  # [B, n_mels, T]
        
        # Add channel dimension back for model input
        return log_mel.unsqueeze(1).half()  # [B, 1, n_mels, T]

    def forward(self, waveform: Tensor, device_ids: List[str]) -> Tensor:
        """
        Perform forward pass for a batch of waveforms.

        Args:
            waveform: Tensor of shape [B, 1, n_samples]
            device_ids: List of device identifiers (length B)

        Returns:
            Tensor of shape [B, n_classes]
        """
        with torch.no_grad():
            mel = self.preprocess(waveform)  # [B, 1, n_mels, T]
            logits = self.model(mel, device_ids)  # [B, n_classes]
        return logits


def load_model(model_file_path: Optional[str] = None) -> Baseline:
    """
    Load the student device-specific fine-tuned model from a checkpoint.

    Args:
        model_file_path: Optional path to a .pt file. If None, uses the default packaged checkpoint.

    Returns:
        A Baseline model instance with loaded weights.
    """
    config = Config()
    model = Baseline(config)

    # Use default checkpoint from package resources if no path is given
    if model_file_path is None:
        with pkg_resources.path(ckpts, "student_device_specific_10.pt") as ckpt_path:
            model_file_path = str(ckpt_path)

    # Load checkpoint to CPU (compatible with CPU inference)
    ckpt = torch.load(model_file_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "device_models" in ckpt:
        # This is our student device-specific checkpoint format
        device_models_state = ckpt["device_models"]
        
        # Load each device-specific model
        for device_id, device_state_dict in device_models_state.items():
            if device_id in model.model.device_models:
                model.model.device_models[device_id].load_state_dict(device_state_dict, strict=True)
                print(f"✅ Loaded device-specific model for device: {device_id}")
        
        # Load original model if available (for unseen devices)
        if "original_model" in ckpt:
            # Direct original model state dict
            model.model.original_model.load_state_dict(ckpt["original_model"], strict=True)
            print("✅ Loaded original pre-trained model for unseen devices")
        elif "state_dict" in ckpt:
            # Extract original model state from full state dict
            original_model_prefix = "multi_device_student.original_model."
            original_state_dict = {}
            
            for key, value in ckpt["state_dict"].items():
                if key.startswith(original_model_prefix):
                    clean_key = key[len(original_model_prefix):]
                    original_state_dict[clean_key] = value
            
            if original_state_dict:
                model.model.original_model.load_state_dict(original_state_dict, strict=True)
                print("✅ Loaded original pre-trained model for unseen devices")
            else:
                print("⚠️  Warning: No original model weights found in checkpoint")
        else:
            print("⚠️  Warning: No original model weights found in checkpoint")
    
    elif "state_dict" in ckpt:
        # Handle Lightning-style checkpoints
        # Extract multi_device_student state
        multi_device_prefix = "multi_device_student."
        multi_device_state = {}
        
        for key, value in ckpt["state_dict"].items():
            if key.startswith(multi_device_prefix):
                clean_key = key[len(multi_device_prefix):]
                multi_device_state[clean_key] = value
        
        if multi_device_state:
            model.model.load_state_dict(multi_device_state, strict=True)
            print("✅ Loaded multi-device student model from Lightning checkpoint")
    
    else:
        # Direct state dict
        model.model.load_state_dict(ckpt, strict=True)
        print("✅ Loaded model from direct state dict")

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
    Load and preprocess audio files in parallel with batch-wise resampling and STFT.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for STFT/mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    assert len(file_paths) == len(device_ids)

    device = next(model.parameters()).device
    target_sr = model.config.sample_rate

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

    # Step 2: Group by original sample rate to batch resampling
    sr_groups = defaultdict(list)
    for idx, waveform, sr in loaded:
        sr_groups[sr].append((idx, waveform))

    inputs: List[Tensor] = [None] * len(file_paths)  # final output buffer

    # Step 3: Batch resample and preprocess each group
    print("Batched Resampling ...")
    for sr, items in sr_groups.items():
        print(f"Processing SR={sr} with {len(items)} files")
        for i in tqdm(range(0, len(items), batch_size), desc=f"SR={sr}", leave=False):
            chunk = items[i:i + batch_size]
            indices, waveforms = zip(*chunk)

            # Pad waveforms to the same length → [max_len, B]
            padded = pad_sequence(waveforms, batch_first=False)

            # Reshape → [B, 1, samples]
            batch_wave = padded.transpose(0, 1).unsqueeze(1).to(device)

            # # Resample if needed
            # if sr != target_sr:
            #     batch_wave = torchaudio.functional.resample(
            #         batch_wave, orig_freq=sr, new_freq=target_sr
            #     )

            # STFT + mel computation for batch
            with torch.no_grad():
                mel_batch = model.preprocess(batch_wave)  # [B, 1, n_mels, T]

            # Store each preprocessed mel in original order
            for mel, idx in zip(mel_batch, indices):
                inputs[idx] = mel.unsqueeze(0).cpu()  # [1, 1, n_mels, T]

    return inputs


def get_model_for_device(
    model: Baseline,
    device_id: str
) -> torch.nn.Module:
    """
    Extract the device model corresponding to a specific device ID.
    For unseen devices, returns the original pre-trained model.

    Args:
        model: Baseline model instance.
        device_id: Device identifier string (e.g., 's1', 's4').

    Returns:
        The device model (nn.Module) associated with the given device.
    """
    device_model = model.model.get_model_for_device(device_id)
    device_model.half()  # ensure float16 to meet complexity constraints
    return device_model


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using device-specific student models.

    Files are grouped by device ID and processed in batches.
    Unseen devices (s4, s5, s6) use the original pre-trained model.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.pt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

    Returns:
        A tuple (logits, class_order)
        - logits: List of tensors, one per file, each of shape [n_classes]
        - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path).to(device)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

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


def predict_with_general_model(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using ONLY the general (original) model.
    This ignores device-specific models and uses the pre-trained original model for all devices.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.pt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

    Returns:
        A tuple (logits, class_order)
        - logits: List of tensors, one per file, each of shape [n_classes]
        - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path).to(device)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

    # Step 2: Use ONLY the original general model for all samples
    general_model = model.model.get_original_model().to(device)
    general_model.eval()
    general_model.half()  # use float16 to meet complexity constraints

    outputs = []  # Final predictions

    # Step 3: Process all samples with the general model in batches
    print("🔄 Running inference with general model only...")
    all_mels = [inp.squeeze(0).squeeze(0) for inp in inputs]  # List of [n_mels, T]
    
    for i in tqdm(range(0, len(all_mels), batch_size), desc="General model inference"):
        chunk_mels = all_mels[i:i + batch_size]
        
        # Pad to same length → [max_T, batch, n_mels]
        padded = pad_sequence([m.T for m in chunk_mels], batch_first=False)

        # Reshape → [batch, 1, n_mels, max_T]
        batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)

        with torch.no_grad():
            logits = general_model(batch).cpu()  # [B, n_classes]

        # Store results
        for logit in logits:
            outputs.append(logit)

    return outputs, model.class_order


def calculate_detailed_accuracy(
    predictions: List[torch.Tensor],
    true_labels: List[int],
    scene_labels: List[str],
    device_ids: List[str],
    class_order: List[str],
    trained_devices: Optional[List[str]] = None,
    verbose: bool = True
) -> dict:
    """
    Calculate detailed accuracy metrics including class-wise and device-wise accuracy.
    
    Args:
        predictions: List of prediction tensors from model
        true_labels: List of true class indices
        scene_labels: List of scene label strings
        device_ids: List of device IDs
        class_order: List of class names in order
        trained_devices: List of devices the model was trained on
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing all accuracy metrics
    """
    from collections import defaultdict
    import numpy as np
    
    # Get predicted labels
    pred_labels = [pred.argmax().item() for pred in predictions]
    
    # Overall accuracy
    overall_acc = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)
    
    # Class-wise accuracy calculation
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true_label, pred_label, scene_label in zip(true_labels, pred_labels, scene_labels):
        class_total[scene_label] += 1
        if true_label == pred_label:
            class_correct[scene_label] += 1
    
    class_accuracies = {}
    for class_name in class_order:
        if class_name in class_total and class_total[class_name] > 0:
            class_acc = class_correct[class_name] / class_total[class_name]
            class_accuracies[class_name] = class_acc
        else:
            class_accuracies[class_name] = 0.0
    
    # Macro-average accuracy
    macro_avg_acc = np.mean(list(class_accuracies.values()))
    
    # Device-wise accuracy calculation
    device_correct = defaultdict(int)
    device_total = defaultdict(int)
    
    for true_label, pred_label, device_id in zip(true_labels, pred_labels, device_ids):
        device_total[device_id] += 1
        if true_label == pred_label:
            device_correct[device_id] += 1
    
    device_accuracies = {}
    for device_id in set(device_ids):
        if device_total[device_id] > 0:
            device_acc = device_correct[device_id] / device_total[device_id]
            device_accuracies[device_id] = device_acc
        else:
            device_accuracies[device_id] = 0.0
    
    # Group devices if trained_devices is provided
    group_accuracies = {}
    if trained_devices is not None:
        all_devices = list(set(device_ids))
        unseen_devices = [d for d in all_devices if d not in trained_devices]
        real_devices = ['a', 'b', 'c']
        seen_sim_devices = [d for d in trained_devices if d not in real_devices]
        unseen_sim_devices = [d for d in unseen_devices if d not in real_devices]
        
        # Calculate group accuracies
        for group_name, group_devices in [
            ('real', real_devices),
            ('seen_simulated', seen_sim_devices),
            ('unseen_simulated', unseen_sim_devices)
        ]:
            group_correct = sum(device_correct[d] for d in group_devices if d in device_correct)
            group_total = sum(device_total[d] for d in group_devices if d in device_total)
            if group_total > 0:
                group_accuracies[group_name] = group_correct / group_total
            else:
                group_accuracies[group_name] = 0.0
    
    # Compile results
    results = {
        'overall_accuracy': overall_acc,
        'macro_average_accuracy': macro_avg_acc,
        'class_wise_accuracy': class_accuracies,
        'device_wise_accuracy': device_accuracies,
        'group_wise_accuracy': group_accuracies,
        'class_counts': dict(class_total),
        'device_counts': dict(device_total)
    }
    
    if verbose:
        print(f"\n=== DETAILED ACCURACY ANALYSIS ===")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Macro-Average Accuracy: {macro_avg_acc:.4f}")
        
        print(f"\n=== CLASS-WISE ACCURACY ===")
        for class_name in class_order:
            acc = class_accuracies[class_name]
            count = class_total.get(class_name, 0)
            correct = class_correct.get(class_name, 0)
            print(f"   {class_name:20s}: {acc:.4f} ({correct:4d}/{count:4d})")
        
        print(f"\n=== DEVICE-WISE ACCURACY ===")
        if trained_devices is not None:
            # Group by device type
            real_devices = ['a', 'b', 'c']
            if any(d in device_accuracies for d in real_devices):
                print(f"📱 REAL DEVICES:")
                for device_id in real_devices:
                    if device_id in device_accuracies:
                        acc = device_accuracies[device_id]
                        count = device_total[device_id]
                        correct = device_correct[device_id]
                        print(f"   Device {device_id}: {acc:.4f} ({correct:4d}/{count:4d})")
                if 'real' in group_accuracies:
                    print(f"   📊 REAL devices average: {group_accuracies['real']:.4f}")
            
            seen_sim_devices = [d for d in trained_devices if d not in real_devices]
            if seen_sim_devices and any(d in device_accuracies for d in seen_sim_devices):
                print(f"\n👁️  SEEN SIMULATED DEVICES:")
                for device_id in seen_sim_devices:
                    if device_id in device_accuracies:
                        acc = device_accuracies[device_id]
                        count = device_total[device_id]
                        correct = device_correct[device_id]
                        print(f"   Device {device_id}: {acc:.4f} ({correct:4d}/{count:4d})")
                if 'seen_simulated' in group_accuracies:
                    print(f"   👁️  SEEN simulated devices average: {group_accuracies['seen_simulated']:.4f}")
            
            all_devices = list(set(device_ids))
            unseen_devices = [d for d in all_devices if d not in trained_devices]
            unseen_sim_devices = [d for d in unseen_devices if d not in real_devices]
            if unseen_sim_devices and any(d in device_accuracies for d in unseen_sim_devices):
                print(f"\n🔍 UNSEEN SIMULATED DEVICES:")
                for device_id in unseen_sim_devices:
                    if device_id in device_accuracies:
                        acc = device_accuracies[device_id]
                        count = device_total[device_id]
                        correct = device_correct[device_id]
                        print(f"   Device {device_id}: {acc:.4f} ({correct:4d}/{count:4d})")
                if 'unseen_simulated' in group_accuracies:
                    print(f"   🎯 UNSEEN simulated devices average: {group_accuracies['unseen_simulated']:.4f}")
                    print(f"\n🔥 KEY METRIC - UNSEEN DEVICE PERFORMANCE: {group_accuracies['unseen_simulated']:.4f}")
        else:
            # Simple device listing if no trained_devices info
            for device_id in sorted(device_accuracies.keys()):
                acc = device_accuracies[device_id]
                count = device_total[device_id]
                correct = device_correct[device_id]
                print(f"   Device {device_id}: {acc:.4f} ({correct:4d}/{count:4d})")
    
    return results


def evaluate_model_performance(
    file_paths: List[str],
    device_ids: List[str],
    scene_labels: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64,
    verbose: bool = True
) -> dict:
    """
    Comprehensive model evaluation with detailed accuracy metrics.
    
    Args:
        file_paths: List of audio file paths
        device_ids: List of device IDs
        scene_labels: List of ground truth scene labels
        model_file_path: Optional path to model checkpoint
        use_cuda: Whether to use GPU
        batch_size: Batch size for inference
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Run predictions
    predictions, class_order = predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=model_file_path,
        use_cuda=use_cuda,
        batch_size=batch_size
    )
    
    # Convert scene labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    true_labels = [label_to_idx[label] for label in scene_labels]
    
    # Load model to get device configuration
    model = load_model(model_file_path)
    trained_devices = model.config.train_device_ids
    
    # Calculate detailed accuracy
    results = calculate_detailed_accuracy(
        predictions=predictions,
        true_labels=true_labels,
        scene_labels=scene_labels,
        device_ids=device_ids,
        class_order=class_order,
        trained_devices=trained_devices,
        verbose=verbose
    )
    
    # Add additional metadata
    results['model_info'] = {
        'trained_devices': trained_devices,
        'all_devices': model.config.all_device_ids,
        'unseen_devices': [d for d in model.config.all_device_ids if d not in trained_devices],
        'class_order': class_order,
        'total_samples': len(file_paths),
        'num_classes': len(class_order),
        'num_devices': len(set(device_ids))
    }
    
    return results
