import gc
import atexit
import copy
import contextlib
import math
import os
import re
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import torch.utils.data
from concurrent.futures import ProcessPoolExecutor

from .config import (
    output_dir,
    image_paths_files,
    gt_paths_file,
    mask_paths_file,
    val_image_paths_files,
    val_gt_paths_file,
    val_mask_paths_file,
    num_classes,
    epochs,
    num_channels,
    slicing_plane,
    training_schedule_file,
    pretrained_model_path,
    pretrained_model_load_optimizer,
    print_every_n_subbatches,
    num_input_slices,
    num_output_slices,
    minimum_height_width,
    use_flip_augmentation,
    use_intensity_augmentation,
    intensity_augmentation_strength,
    use_rotation_augmentation,
    rotation_degrees,
    use_deep_supervision,
    deep_supervision_weights,
    use_mixed_precision,
    mixed_precision_dtype,
    use_brainiac_embeddings,
    brainiac_embedding_type,
    brainiac_feature_paths_files,
    val_brainiac_feature_paths_files,
    channel_names,
    optional_channels,
    channel_dropout_probabilities,
    channel_dropout_strategy,
    channel_dropout_subset_probabilities,
    use_ema,
    ema_decay,
    use_dinov3_embeddings,
    dinov3_frozen_epochs,
)
from .data_loading import (
    read_paths_from_file,
    detect_input_shape,
    load_epoch_dataset,
    load_val_dataset,
    compute_class_weights_from_dataset,
    AstrilSliceDataset,
    _dataloader_worker_init,
)
from .model_architecture import (
    create_dynamic_unet_from_config,
    create_dynamic_unet_from_metadata,
)
from .utils import (
    init_logging,
    parse_and_validate_schedule_params,
    combined_focal_tversky_wce_loss,
    append_metrics_to_file,
    append_training_stats,
    get_vram_stats_mb,
    get_latest_checkpoint,
    get_epoch_from_checkpoint,
    get_parameters_for_epoch,
    compute_masked_predictions,
    compute_weighted_macro_metrics,
)


def get_checkpoint_name(epoch: int) -> str:
    return f"epoch_{epoch}.pt"


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_input_tensor(x_batch, device):
    if torch.is_tensor(x_batch):
        x = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        x = torch.as_tensor(x_batch, dtype=torch.float32).to(device=device, non_blocking=True)
    return x.permute(0, 3, 1, 2).contiguous()


def _to_brainiac_tensor(b_batch, device):
    if torch.is_tensor(b_batch):
        b = b_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        b = torch.as_tensor(b_batch, dtype=torch.float32).to(device=device, non_blocking=True)
    return b.permute(0, 3, 1, 2).contiguous()


def _to_target_tensors(y_batch, mask_batch, device):
    if torch.is_tensor(y_batch):
        y = y_batch.to(device=device, dtype=torch.long, non_blocking=True)
    else:
        y = torch.as_tensor(y_batch, dtype=torch.long).to(device=device, non_blocking=True)
    if torch.is_tensor(mask_batch):
        mask = mask_batch.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        mask = torch.as_tensor(mask_batch, dtype=torch.float32).to(device=device, non_blocking=True)
    y_onehot = F.one_hot(y.squeeze(-1), num_classes=num_classes).to(dtype=torch.float32)
    return y, y_onehot, mask


def _tensor_params(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _empty_metric_accumulators():
    return {
        "loss_sum": np.zeros(num_classes, dtype=np.float64),
        "loss_count": np.zeros(num_classes, dtype=np.int64),
        "loss_all_sum": 0.0,
        "loss_all_count": 0,
        "correct_by_class": np.zeros(num_classes, dtype=np.int64),
        "gt_count_by_class": np.zeros(num_classes, dtype=np.int64),
        "pred_count_by_class": np.zeros(num_classes, dtype=np.int64),
        "total_samples": 0,
    }


def _update_prediction_metrics(acc, probabilities, y_batch, mask_batch):
    pred_filtered, gt_filtered = compute_masked_predictions(
        probabilities, y_batch, mask_batch, num_classes
    )
    acc["total_samples"] += len(gt_filtered)
    for c in range(num_classes):
        pred_c = pred_filtered == c
        gt_c = gt_filtered == c
        acc["correct_by_class"][c] += np.sum(pred_c & gt_c)
        acc["gt_count_by_class"][c] += np.sum(gt_c)
        acc["pred_count_by_class"][c] += np.sum(pred_c)


def _probabilities_for_metrics(logits):
    """Convert logits to NumPy probabilities through FP32 for BF16 compatibility."""
    return F.softmax(logits.detach().float(), dim=-1).cpu().numpy()


def _update_loss_metrics(acc, loss_value, loss_per_class):
    acc["loss_all_sum"] += float(loss_value)
    acc["loss_all_count"] += 1
    acc["loss_sum"] += loss_per_class.detach().cpu().numpy()
    acc["loss_count"] += 1


def _metrics_for_logging(acc):
    weighted = compute_weighted_macro_metrics(acc, num_classes)
    class_metrics = {}
    for c in range(num_classes):
        tp = acc["correct_by_class"][c]
        gt = acc["gt_count_by_class"][c]
        pred = acc["pred_count_by_class"][c]
        fp = pred - tp
        fn = gt - tp
        tn = acc["total_samples"] - tp - fp - fn
        denom_iou = tp + fp + fn
        class_metrics[c] = {
            "iou": tp / float(denom_iou + 1e-9),
            "accuracy": (tp + tn) / float(acc["total_samples"] + 1e-9),
            "precision": tp / float(pred + 1e-9),
            "recall": tp / float(gt + 1e-9),
            "loss": acc["loss_sum"][c] / float(max(acc["loss_count"][c], 1)),
        }
    all_classes_metrics = {
        "accuracy": weighted["micro_accuracy"],
        "precision": weighted["weighted_macro_precision"],
        "recall": weighted["weighted_macro_recall"],
        "loss": acc["loss_all_sum"] / float(max(acc["loss_all_count"], 1)),
    }
    return class_metrics, all_classes_metrics


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensors_are_finite(value):
    """Return False if any tensor nested in a checkpoint contains NaN/Inf."""
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tensors_are_finite(item) for item in value)
    return True


class ExponentialMovingAverage:
    """Maintain a detached model copy for stable validation and inference."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.model.named_parameters():
            source = source_parameters[name].detach()
            ema_parameter.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(source_buffers[name].detach())

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def _checkpoint_is_valid(path, device):
    try:
        checkpoint = _load_checkpoint(path, device)
        return (
            isinstance(checkpoint, dict)
            and _tensors_are_finite(checkpoint.get("model_state_dict", {}))
            and _tensors_are_finite(checkpoint.get("ema_model_state_dict", {}))
            and _tensors_are_finite(checkpoint.get("optimizer_state_dict", {}))
        )
    except Exception as exc:
        print(f"WARNING: Could not validate checkpoint {path}: {exc}")
        return False


def _latest_valid_checkpoint(checkpoint_dir, device):
    candidates = []
    for name in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []:
        match = re.match(r"^epoch_(\d+)\.pt$", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))
    for epoch, path in sorted(candidates, reverse=True):
        if _checkpoint_is_valid(path, device):
            return path
        print(f"WARNING: Ignoring non-finite or unreadable checkpoint {path} (epoch {epoch}).")
    return None


def _model_parameters_are_finite(model):
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _accumulation_window_size(batch_counter, total_batches, accumulate_n_sub_batches):
    window_start = ((batch_counter - 1) // accumulate_n_sub_batches) * accumulate_n_sub_batches + 1
    window_end = min(window_start + accumulate_n_sub_batches - 1, total_batches)
    return window_end - window_start + 1


def _save_checkpoint(path, model, optimizer, epoch, ema=None):
    checkpoint = {
        "epoch": epoch,
        "architecture": model.architecture_config(),
        "channel_metadata": {
            "channel_names": list(channel_names or []),
            "optional_channels": list(optional_channels or []),
            "channel_dropout_probabilities": dict(channel_dropout_probabilities or {}),
            "channel_dropout_strategy": channel_dropout_strategy,
            "channel_dropout_subset_probabilities": dict(
                channel_dropout_subset_probabilities or {}
            ),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if ema is not None:
        checkpoint["ema_model_state_dict"] = ema.state_dict()
        checkpoint["ema_decay"] = ema.decay
    if not _tensors_are_finite(checkpoint):
        raise RuntimeError(f"Refusing to save non-finite checkpoint for epoch {epoch}.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".checkpoint_", suffix=".pt", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_model_from_checkpoint(path, device):
    checkpoint = _load_checkpoint(path, device)
    model = create_dynamic_unet_from_metadata(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    epoch = int(checkpoint.get("epoch") or get_epoch_from_checkpoint(path) or 0)
    return model, checkpoint, epoch


def _resolve_mixed_precision(device, enabled, requested_dtype):
    if not enabled or device.type != "cuda":
        return False, None, "disabled"

    requested = str(requested_dtype or "auto").strip().lower()
    if requested not in {"auto", "bf16", "fp16"}:
        raise ValueError(
            "mixed_precision_dtype must be one of: auto, bf16, fp16; "
            f"got {requested_dtype!r}"
        )
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 mixed precision was requested but is not supported by this CUDA device.")
        return True, torch.bfloat16, "bf16"
    return True, torch.float16, "fp16"


def _autocast_context(device, enabled, dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _sample_names_text(sample_names, limit=5):
    if isinstance(sample_names, str):
        names = [sample_names]
    elif isinstance(sample_names, (list, tuple)):
        names = [str(name) for name in sample_names]
    else:
        names = [str(sample_names)]
    suffix = ", ..." if len(names) > limit else ""
    return ", ".join(names[:limit]) + suffix


def _tensor_range(tensor):
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return "no finite values"
    return f"min={finite.min().item():.6g}, max={finite.max().item():.6g}"


def _validate_batch_inputs(x, brainiac, mask, sample_names, context):
    tensors = [("input", x), ("mask", mask)]
    if brainiac is not None:
        tensors.append(("brainiac", brainiac))
    invalid = [name for name, tensor in tensors if not torch.isfinite(tensor).all().item()]
    if invalid:
        summaries = "; ".join(
            f"{name}: {_tensor_range(tensor)}" for name, tensor in tensors
        )
        raise FloatingPointError(
            f"Non-finite {context} batch input(s) {invalid}; "
            f"samples=[{_sample_names_text(sample_names)}]; {summaries}"
        )


def _resize_channel_last_spatial(tensor, spatial_size):
    """Nearest-neighbor resize for (N,H,W,S,C) targets and masks."""
    if tuple(tensor.shape[1:3]) == tuple(spatial_size):
        return tensor
    n, _, _, slices, channels = tensor.shape
    channel_first = tensor.permute(0, 3, 4, 1, 2).reshape(
        n, slices * channels, tensor.shape[1], tensor.shape[2]
    )
    resized = F.interpolate(channel_first.float(), size=spatial_size, mode="nearest")
    return resized.view(n, slices, channels, *spatial_size).permute(0, 3, 4, 1, 2).contiguous()


def _compute_model_loss(
    model,
    x,
    brainiac,
    y_onehot,
    mask,
    class_weights,
    alpha,
    beta,
    tversky_gamma,
    wce_weight,
    label_smoothing,
    use_brainiac_fusion,
    use_deep_supervision,
    ds_loss_weight,
):
    output = model(x, brainiac) if use_brainiac_fusion else model(x)
    if use_deep_supervision:
        logits, aux_outputs = output
    else:
        logits, aux_outputs = output, []
    loss_value, loss_per_class = combined_focal_tversky_wce_loss(
        y_onehot,
        logits,
        mask,
        class_weights,
        alpha,
        beta,
        gamma=tversky_gamma,
        wce_weight=wce_weight,
        label_smoothing=label_smoothing,
    )
    if use_deep_supervision and aux_outputs:
        ds_weights = deep_supervision_weights if deep_supervision_weights else [0.25, 0.125]
        for i, aux in enumerate(aux_outputs):
            aux_targets = _resize_channel_last_spatial(y_onehot, aux.shape[1:3])
            aux_mask = _resize_channel_last_spatial(mask, aux.shape[1:3])
            aux_loss, _ = combined_focal_tversky_wce_loss(
                aux_targets,
                aux,
                aux_mask,
                class_weights,
                alpha,
                beta,
                gamma=tversky_gamma,
                wce_weight=wce_weight,
                label_smoothing=label_smoothing,
            )
            weight = ds_weights[i] if i < len(ds_weights) else ds_weights[-1]
            loss_value = loss_value + ds_loss_weight * weight * aux_loss
    return logits, loss_value, loss_per_class


def _cleanup_temp_dirs(temp_dirs):
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


def _reset_temp_base_dir(temp_base_dir):
    if os.path.exists(temp_base_dir):
        print(f"Removing stale training temp cache: {temp_base_dir}")
        shutil.rmtree(temp_base_dir, ignore_errors=True)
    os.makedirs(temp_base_dir, exist_ok=True)


def _loader_kwargs(num_workers, prefetch_factor, persistent_workers, device):
    num_workers = max(int(num_workers or 0), 0)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "worker_init_fn": _dataloader_worker_init,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(int(prefetch_factor or 2), 1)
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs


def train_model():
    log_file, log_file_path = init_logging(output_dir)
    print(f"Logging to: {log_file_path}")

    device = _select_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    training_schedule = pd.read_csv(training_schedule_file, sep="\t")
    optional_channel_indices = [
        channel_names.index(name) for name in optional_channels if name in channel_names
    ]
    dropout_by_index = {
        channel_names.index(name): prob
        for name, prob in dict(channel_dropout_probabilities or {}).items()
        if name in channel_names
    }
    brainiac_channel_indices = None

    train_channel_paths = [read_paths_from_file(path) for path in image_paths_files]
    assert all(len(train_channel_paths[0]) == len(p) for p in train_channel_paths), \
        "Mismatch in the number of paths across training channels"
    train_volume_paths_list = [list(scan) for scan in zip(*train_channel_paths)]
    train_mask_paths = read_paths_from_file(mask_paths_file)
    train_gt_paths = read_paths_from_file(gt_paths_file)

    val_channel_paths = [read_paths_from_file(path) for path in val_image_paths_files]
    assert all(len(val_channel_paths[0]) == len(p) for p in val_channel_paths), \
        "Mismatch in the number of paths across validation channels"
    val_volume_paths_list = [list(scan) for scan in zip(*val_channel_paths)]
    val_mask_paths = read_paths_from_file(val_mask_paths_file)
    val_gt_paths = read_paths_from_file(val_gt_paths_file)

    if use_brainiac_embeddings and brainiac_embedding_type != "encoder_fusion":
        raise ValueError(
            "BrainIAC now supports only brainiac_embedding_type = encoder_fusion. "
            f"Found: {brainiac_embedding_type}"
        )
    use_brainiac_fusion = bool(use_brainiac_embeddings)
    train_brainiac_paths_list = None
    val_brainiac_paths_list = None
    if use_brainiac_fusion:
        if not brainiac_feature_paths_files or not val_brainiac_feature_paths_files:
            raise ValueError("BrainIAC encoder-fusion requires train and validation BrainIAC feature path cfg files.")
        train_brainiac_paths = [read_paths_from_file(path) for path in brainiac_feature_paths_files]
        val_brainiac_paths = [read_paths_from_file(path) for path in val_brainiac_feature_paths_files]
        assert all(len(train_brainiac_paths[0]) == len(p) for p in train_brainiac_paths), \
            "Mismatch in the number of paths across training BrainIAC feature channels"
        assert all(len(val_brainiac_paths[0]) == len(p) for p in val_brainiac_paths), \
            "Mismatch in the number of paths across validation BrainIAC feature channels"
        train_brainiac_paths_list = [list(scan) for scan in zip(*train_brainiac_paths)]
        val_brainiac_paths_list = [list(scan) for scan in zip(*val_brainiac_paths)]
        from .config import brainiac_encode_channel_indices
        brainiac_channel_indices = list(brainiac_encode_channel_indices or range(len(train_brainiac_paths)))
        print(f"BrainIAC encoder fusion enabled with {len(train_brainiac_paths)} embedding sources.")

    input_shape, original_shape = detect_input_shape(
        sample_file_path=train_mask_paths[0],
        slicing_plane=slicing_plane,
        num_channels=num_channels,
    )
    print(f"Original Shape: {original_shape}")
    print(f"Padded Input Shape: {input_shape}")

    model_subdir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_subdir, exist_ok=True)
    latest_checkpoint = _latest_valid_checkpoint(model_subdir, device)

    optimizer_checkpoint = None
    if latest_checkpoint:
        starting_epoch = get_epoch_from_checkpoint(latest_checkpoint)
        if starting_epoch is not None and starting_epoch >= epochs:
            print(f"Training completed up to epoch {starting_epoch}. Nothing more to do.")
            log_file.close()
            return
        print(f"Loading PyTorch checkpoint from {latest_checkpoint}...")
        model, optimizer_checkpoint, starting_epoch = _load_model_from_checkpoint(
            latest_checkpoint, device
        )
    elif pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        if not str(pretrained_model_path).endswith(".pt"):
            raise ValueError(
                "pretrained_model_path must point to a PyTorch .pt checkpoint after the migration."
            )
        print(f"Starting from pre-trained PyTorch checkpoint: {pretrained_model_path}")
        model, pretrained_checkpoint, _ = _load_model_from_checkpoint(pretrained_model_path, device)
        if pretrained_model_load_optimizer:
            optimizer_checkpoint = pretrained_checkpoint
            print("Restoring optimizer and EMA state from the pre-trained checkpoint.")
        else:
            optimizer_checkpoint = None
            print("Using pre-trained model weights only; optimizer and EMA state are freshly initialized.")
        starting_epoch = 0
    else:
        print("Creating PyTorch model from scratch.")
        model = create_dynamic_unet_from_config().to(device)
        starting_epoch = 0

    # Track whether DINOv3 has been unfrozen during this training run
    _dinov3_unfrozen = use_dinov3_embeddings and not getattr(model, "dinov3_frozen", True)
    _dinov3_lr_scale = 0.1  # relative LR for DINOv3 param group vs main schedule

    if _dinov3_unfrozen and hasattr(model, "dinov3"):
        # DINOv3 was already unfrozen in a prior run. Recreate the same two-group optimizer
        # structure that was saved in the checkpoint (group 0: non-DINOv3, group 1: DINOv3).
        dinov3_param_ids = {id(p) for p in model.dinov3.parameters()}
        main_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in dinov3_param_ids
        ]
        dinov3_params = [p for p in model.dinov3.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": main_params},
            {"params": dinov3_params},
        ])
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params)

    if optimizer_checkpoint and "optimizer_state_dict" in optimizer_checkpoint:
        optimizer.load_state_dict(optimizer_checkpoint["optimizer_state_dict"])

    ema = ExponentialMovingAverage(model, decay=ema_decay) if use_ema else None
    if ema is not None and optimizer_checkpoint:
        ema_state = optimizer_checkpoint.get("ema_model_state_dict")
        if ema_state:
            ema.load_state_dict(ema_state)
            print(f"Restored EMA weights (decay={ema.decay:g}).")
    if ema is not None and not optimizer_checkpoint:
        print(f"EMA validation/inference weights enabled (decay={ema.decay:g}).")

    use_amp, amp_dtype, amp_dtype_name = _resolve_mixed_precision(
        device, use_mixed_precision, mixed_precision_dtype
    )
    use_grad_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)
    print(
        f"Mixed precision enabled: {use_amp}"
        + (f" (dtype={amp_dtype_name}, grad_scaler={use_grad_scaler})" if use_amp else "")
    )

    model_input_channels = int(
        getattr(model, "expected_input_channels", num_channels * num_input_slices)
    )
    append_modality_presence = bool(getattr(model, "uses_modality_presence", False))
    dummy_input_shape = (1, model_input_channels, minimum_height_width, minimum_height_width)
    with torch.no_grad():
        dummy_x = torch.zeros(dummy_input_shape, dtype=torch.float32, device=device)
        if use_brainiac_fusion:
            brainiac_ch = model.architecture_config()["brainiac_input_channels"]
            dummy_b = torch.zeros(
                (1, brainiac_ch, max(1, minimum_height_width // 16), max(1, minimum_height_width // 16)),
                dtype=torch.float32,
                device=device,
            )
            _ = model(dummy_x, dummy_b)
        else:
            _ = model(dummy_x)
    print("Model built with dummy input shape:", dummy_input_shape)

    train_indexes = np.arange(len(train_mask_paths))
    val_indexes = np.arange(len(val_mask_paths))

    train_metrics_file_path = os.path.join(output_dir, "train_metrics.tsv")
    val_metrics_file_path = os.path.join(output_dir, "val_metrics.tsv")
    training_stats_file_path = os.path.join(output_dir, "training_stats.tsv")

    from .config import (
        n_cores,
        dataloader_num_workers,
        dataloader_prefetch_factor,
        dataloader_persistent_workers,
    )
    _worker_pool = ProcessPoolExecutor(max_workers=n_cores)

    # Temp files live alongside the output directory so they stay on the same drive
    # as the input data, avoiding slow cross-drive writes via system %TEMP%.
    _temp_base_dir = os.path.join(output_dir, "_astril_temp")
    _reset_temp_base_dir(_temp_base_dir)

    # Streaming state: temp dirs + dataset are owned here and cleaned up on reload/exit
    _train_temp_dirs = []
    _train_dataset = None
    train_loader = None
    epoch_class_weights = None

    def _cleanup_training_temp_cache():
        _cleanup_temp_dirs(_train_temp_dirs)
        shutil.rmtree(_temp_base_dir, ignore_errors=True)

    atexit.register(_cleanup_training_temp_cache)

    data_loading_counter = 0
    for epoch in range(starting_epoch, epochs):
        print("\n############################")
        print(f"Epoch {epoch+1}/{epochs}")
        print("############################")

        current_params = get_parameters_for_epoch(epoch + 1, training_schedule)
        try:
            parsed = parse_and_validate_schedule_params(
                current_params=current_params,
                num_classes=num_classes,
                train_indexes=train_indexes,
            )
        except ValueError as e:
            print(f"ERROR in schedule parameters for epoch {epoch+1}: {e}")
            break

        scan_batch_size = parsed["scan_batch_size"]
        slice_sub_batch_size = parsed["slice_sub_batch_size"]
        accumulate_n_sub_batches = parsed["accumulate_n_sub_batches"]
        conduct_validation = parsed["conduct_validation"]
        validation_frequency = parsed["validation_frequency"]
        learning_rate = parsed["learning_rate"]
        wce_weight = parsed["wce_weight"]
        tversky_gamma = parsed["tversky_gamma"]
        class_weights = parsed["class_weights"]
        epochs_per_data = parsed["epochs_per_data"]
        class_multiplication_factors = parsed["class_multiplication_factors"]
        require_classes = parsed["require_classes"]
        alpha_vals_list = parsed["alpha_vals_list"]
        beta_vals_list = parsed["beta_vals_list"]
        gradient_clip_norm = parsed["gradient_clip_norm"]
        label_smoothing = parsed["label_smoothing"]
        ds_loss_weight = parsed["deep_supervision_loss_weight"]

        for i, group in enumerate(optimizer.param_groups):
            # Last param group is DINOv3 backbone (added after unfreeze) — keep at relative scale
            if _dinov3_unfrozen and i == len(optimizer.param_groups) - 1:
                group["lr"] = learning_rate * _dinov3_lr_scale
            else:
                group["lr"] = learning_rate
        print(f"  Effective LR: {learning_rate:.6f}")

        # --- DINOv3 selective unfreeze ---
        if (
            use_dinov3_embeddings
            and not _dinov3_unfrozen
            and dinov3_frozen_epochs is not None
            and (epoch + 1) > dinov3_frozen_epochs
            and hasattr(model, "set_dinov3_frozen")
        ):
            model.set_dinov3_frozen(False)
            _dinov3_unfrozen = True
            dinov3_finetune_lr = learning_rate * 0.1
            _dinov3_lr_scale = 0.1
            optimizer.add_param_group({
                "params": list(model.dinov3.parameters()),
                "lr": dinov3_finetune_lr,
            })
            print(
                f"  DINOv3 backbone unfrozen at epoch {epoch + 1}. "
                f"Added to optimizer with LR={dinov3_finetune_lr:.2e} ({_dinov3_lr_scale:.0%} of main LR)"
            )

        # --- Data loading / reuse decision ---
        data_loading_counter += 1
        need_reload = _train_dataset is None or (data_loading_counter % epochs_per_data == 0)

        if need_reload:
            data_loading_counter = 0
            # Join DataLoader workers before deleting temp files (Windows file locking)
            if train_loader is not None:
                del train_loader
                train_loader = None
                gc.collect()
            _cleanup_temp_dirs(_train_temp_dirs)
            _train_temp_dirs = []
            _train_dataset = None

            print("Loading training volumes for this epoch...")
            mem = psutil.virtual_memory()
            print(f"System RAM usage before data load: {mem.percent:.2f}%")
            t0_data = time.perf_counter()

            selected_train_indexes = np.random.choice(train_indexes, scan_batch_size, replace=False)
            _train_dataset, _train_temp_dirs = load_epoch_dataset(
                scan_indexes=selected_train_indexes,
                volume_paths_list=train_volume_paths_list,
                mask_paths=train_mask_paths,
                gt_paths=train_gt_paths,
                slicing_plane=slicing_plane,
                num_input_slices=num_input_slices,
                num_output_slices=num_output_slices,
                class_multiplication_factors=class_multiplication_factors,
                require_classes=require_classes,
                use_flip_augmentation=use_flip_augmentation,
                use_intensity_augmentation=use_intensity_augmentation,
                intensity_augmentation_strength=intensity_augmentation_strength,
                use_rotation_augmentation=use_rotation_augmentation,
                rotation_degrees=rotation_degrees,
                brainiac_paths_list=train_brainiac_paths_list if use_brainiac_fusion else None,
                target_height=minimum_height_width,
                target_width=minimum_height_width,
                executor=_worker_pool,
                temp_base_dir=_temp_base_dir,
                optional_channel_indices=optional_channel_indices,
                channel_dropout_probabilities=dropout_by_index,
                channel_dropout_strategy=channel_dropout_strategy,
                channel_dropout_subset_probabilities=channel_dropout_subset_probabilities,
                append_modality_presence=append_modality_presence,
                brainiac_channel_indices=brainiac_channel_indices,
            )
            epoch_class_weights = compute_class_weights_from_dataset(_train_dataset, num_classes)

            data_load_s = time.perf_counter() - t0_data
            print(f"Volumes loaded in {data_load_s:.1f}s. Slice entries: {len(_train_dataset)}.")
            mem = psutil.virtual_memory()
            print(f"System RAM usage after data load: {mem.percent:.2f}%")

        else:
            # Reuse same temp dirs — rebuild Dataset cheaply (re-runs _build_index with same volumes)
            _train_dataset = AstrilSliceDataset(
                scan_temp_dirs=_train_temp_dirs,
                num_input_slices=num_input_slices,
                num_output_slices=num_output_slices,
                class_multiplication_factors=class_multiplication_factors,
                require_classes=require_classes,
                is_training=True,
                use_flip_augmentation=use_flip_augmentation,
                use_intensity_augmentation=use_intensity_augmentation,
                intensity_augmentation_strength=intensity_augmentation_strength,
                use_rotation_augmentation=use_rotation_augmentation,
                rotation_degrees=rotation_degrees,
                has_brainiac=use_brainiac_fusion,
                optional_channel_indices=optional_channel_indices,
                channel_dropout_probabilities=dropout_by_index,
                channel_dropout_strategy=channel_dropout_strategy,
                channel_dropout_subset_probabilities=channel_dropout_subset_probabilities,
                append_modality_presence=append_modality_presence,
                brainiac_channel_indices=brainiac_channel_indices,
            )
            epoch_class_weights = compute_class_weights_from_dataset(_train_dataset, num_classes)
            data_load_s = 0.0

        # --- Class weights ---
        if class_weights is not None:
            final_class_weights = [
                epoch_class_weights[i] if math.isnan(class_weights[i]) else class_weights[i]
                for i in range(num_classes)
            ]
            print(f"Using mixed user+dynamic class weights: {final_class_weights}")
        else:
            final_class_weights = epoch_class_weights
            print(f"Using dynamic class weights: {final_class_weights}")

        cw_t    = _tensor_params(final_class_weights, device)
        alpha_t = _tensor_params(alpha_vals_list, device)
        beta_t  = _tensor_params(beta_vals_list, device)

        # --- DataLoader ---
        print(
            "DataLoader settings: "
            f"workers={dataloader_num_workers}, prefetch={dataloader_prefetch_factor}, "
            f"persistent={bool(dataloader_persistent_workers and dataloader_num_workers > 0)}"
        )
        train_loader = torch.utils.data.DataLoader(
            _train_dataset,
            batch_size=slice_sub_batch_size,
            shuffle=True,
            drop_last=False,
            **_loader_kwargs(
                dataloader_num_workers,
                dataloader_prefetch_factor,
                dataloader_persistent_workers,
                device,
            ),
        )
        total_batches = len(train_loader)

        print("Training model...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_acc = _empty_metric_accumulators()
        t0_train = time.perf_counter()
        total_train_slices = 0
        grad_norms = []
        dataloader_wait_times = []
        batch_compute_times = []
        epoch_amp_enabled = use_amp
        amp_fallback_count = 0

        train_iter = iter(train_loader)
        next_batch_t0 = time.perf_counter()
        for batch_counter in range(1, total_batches + 1):
            batch = next(train_iter)
            batch_t0 = time.perf_counter()
            dataloader_wait_times.append(batch_t0 - next_batch_t0)
            # Unpack: (X, B_or_sentinel, Y, M, sample_name)
            x_cpu, b_cpu, y_cpu, mask_cpu, sample_names = batch
            y_batch_np = y_cpu.numpy() if torch.is_tensor(y_cpu) else np.asarray(y_cpu)
            if torch.is_tensor(mask_cpu):
                mask_batch_np = mask_cpu.numpy().astype(np.float32)
            else:
                mask_batch_np = np.asarray(mask_cpu, dtype=np.float32)

            x_batch = _to_input_tensor(x_cpu, device)
            if use_brainiac_fusion:
                b_batch = _to_brainiac_tensor(b_cpu, device)
            else:
                b_batch = None
            y_batch, y_onehot, mask_batch = _to_target_tensors(y_cpu, mask_cpu, device)
            total_train_slices += int(x_batch.shape[0])
            _validate_batch_inputs(
                x_batch, b_batch, mask_batch, sample_names, context="training"
            )

            accumulation_window_size = _accumulation_window_size(
                batch_counter, total_batches, accumulate_n_sub_batches
            )

            with _autocast_context(device, epoch_amp_enabled, amp_dtype):
                logits, loss_value, loss_per_class = _compute_model_loss(
                    model,
                    x_batch,
                    b_batch,
                    y_onehot,
                    mask_batch,
                    cw_t,
                    alpha_t,
                    beta_t,
                    tversky_gamma,
                    wce_weight,
                    label_smoothing,
                    use_brainiac_fusion,
                    use_deep_supervision,
                    ds_loss_weight,
                )

            outputs_finite = (
                torch.isfinite(logits).all().item()
                and torch.isfinite(loss_value).all().item()
                and torch.isfinite(loss_per_class).all().item()
            )
            if not outputs_finite and epoch_amp_enabled:
                amp_fallback_count += 1
                print(
                    f"WARNING: Non-finite {amp_dtype_name} output at epoch {epoch + 1}, "
                    f"batch {batch_counter}/{total_batches}; retrying in FP32; "
                    f"samples=[{_sample_names_text(sample_names)}]"
                )
                with _autocast_context(device, False, None):
                    logits, loss_value, loss_per_class = _compute_model_loss(
                        model,
                        x_batch,
                        b_batch,
                        y_onehot,
                        mask_batch,
                        cw_t,
                        alpha_t,
                        beta_t,
                        tversky_gamma,
                        wce_weight,
                        label_smoothing,
                        use_brainiac_fusion,
                        use_deep_supervision,
                        ds_loss_weight,
                    )
                outputs_finite = (
                    torch.isfinite(logits).all().item()
                    and torch.isfinite(loss_value).all().item()
                    and torch.isfinite(loss_per_class).all().item()
                )
                if outputs_finite and amp_fallback_count >= 3:
                    epoch_amp_enabled = False
                    print(
                        f"WARNING: Disabling autocast for the remainder of epoch {epoch + 1} "
                        f"after {amp_fallback_count} successful FP32 fallbacks."
                    )

            if not outputs_finite:
                raise FloatingPointError(
                    f"Non-finite training output at epoch {epoch + 1}, batch {batch_counter}/{total_batches}; "
                    f"learning_rate={learning_rate:.6g}, loss={loss_value.detach().float().item()}, "
                    f"logits={_tensor_range(logits)}, input={_tensor_range(x_batch)}, "
                    f"samples=[{_sample_names_text(sample_names)}]"
                )

            # Average each accumulation window, including a shorter final window.
            scaled_loss_value = loss_value / float(accumulation_window_size)
            scaler.scale(scaled_loss_value).backward()
            _update_loss_metrics(train_acc, loss_value.detach().float().cpu(), loss_per_class.detach().float())

            is_update_batch = (batch_counter % accumulate_n_sub_batches == 0) or (batch_counter == total_batches)
            if is_update_batch:
                scaler.unscale_(optimizer)
                nonfinite_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                ]
                if nonfinite_gradients:
                    raise FloatingPointError(
                        f"Non-finite gradient at epoch {epoch + 1}, batch {batch_counter}/{total_batches}; "
                        f"learning_rate={learning_rate:.6g}, parameters={nonfinite_gradients[:5]}"
                    )
                if gradient_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf"), error_if_nonfinite=True
                    )
                scaler.step(optimizer)
                scaler.update()
                if not _model_parameters_are_finite(model):
                    raise FloatingPointError(
                        f"Non-finite model parameter after update at epoch {epoch + 1}, "
                        f"batch {batch_counter}/{total_batches}; learning_rate={learning_rate:.6g}"
                    )
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                grad_norms.append(float(grad_norm.detach().cpu()))

                probabilities = _probabilities_for_metrics(logits)
                _update_prediction_metrics(train_acc, probabilities, y_batch_np, mask_batch_np)

            if (batch_counter % print_every_n_subbatches == 0) or (batch_counter == total_batches):
                print(f"Completed {batch_counter}/{total_batches} training batches.")
            batch_compute_times.append(time.perf_counter() - batch_t0)
            next_batch_t0 = time.perf_counter()

        train_s = time.perf_counter() - t0_train
        slices_per_sec = total_train_slices / max(train_s, 1e-6)
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        mean_dataloader_wait_s = float(np.mean(dataloader_wait_times)) if dataloader_wait_times else 0.0
        mean_batch_compute_s = float(np.mean(batch_compute_times)) if batch_compute_times else 0.0
        vram_used_mb, vram_peak_mb = get_vram_stats_mb()
        print(
            f"Epoch {epoch+1} training: {train_s:.1f}s, {slices_per_sec:.0f} slices/s, "
            f"grad_norm={mean_grad_norm:.4f}, VRAM={vram_used_mb:.0f}/{vram_peak_mb:.0f} MB, "
            f"loader_wait={mean_dataloader_wait_s:.4f}s, batch_compute={mean_batch_compute_s:.4f}s"
        )

        checkpoint_filename = get_checkpoint_name(epoch + 1)
        model_file_path = os.path.join(model_subdir, checkpoint_filename)
        _save_checkpoint(model_file_path, model, optimizer, epoch + 1, ema=ema)
        print(f"Saved PyTorch checkpoint to {model_file_path}")

        train_class_metrics, train_all_classes_metrics = _metrics_for_logging(train_acc)
        append_metrics_to_file(
            train_metrics_file_path,
            epoch + 1,
            train_class_metrics,
            all_classes_metrics=train_all_classes_metrics,
        )
        print(f"\nEpoch {epoch+1} Train Report:")
        for class_index, metrics in train_class_metrics.items():
            print(
                f"Class {class_index} - IoU: {metrics['iou']:.3f}, Acc: {metrics['accuracy']:.3f}, "
                f"Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}, "
                f"Loss: {metrics['loss']:.4f}"
            )
        print(
            "ALL_CLASSES (TRAIN) - MicroAcc: {0:.3f}, W-Prec: {1:.3f}, W-Rec: {2:.3f}, Loss: {3:.4f}".format(
                train_all_classes_metrics["accuracy"],
                train_all_classes_metrics["precision"],
                train_all_classes_metrics["recall"],
                train_all_classes_metrics["loss"],
            )
        )

        val_s = None
        if conduct_validation and ((epoch + 1) % validation_frequency == 0):
            print("Conducting validation (2.5D)...")
            t0_val = time.perf_counter()
            val_acc = _empty_metric_accumulators()
            validation_model = ema.model if ema is not None else model
            validation_model.eval()

            val_dataset, val_temp_dirs = load_val_dataset(
                scan_indexes=val_indexes,
                volume_paths_list=val_volume_paths_list,
                mask_paths=val_mask_paths,
                gt_paths=val_gt_paths,
                slicing_plane=slicing_plane,
                num_input_slices=num_input_slices,
                num_output_slices=num_output_slices,
                brainiac_paths_list=val_brainiac_paths_list if use_brainiac_fusion else None,
                target_height=minimum_height_width,
                target_width=minimum_height_width,
                executor=_worker_pool,
                temp_base_dir=_temp_base_dir,
                optional_channel_indices=optional_channel_indices,
                append_modality_presence=append_modality_presence,
                brainiac_channel_indices=brainiac_channel_indices,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=slice_sub_batch_size,
                shuffle=False,
                **_loader_kwargs(
                    dataloader_num_workers,
                    dataloader_prefetch_factor,
                    dataloader_persistent_workers,
                    device,
                ),
            )

            val_amp_enabled = use_amp
            val_amp_fallback_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    x_cpu, b_cpu, y_cpu, mask_cpu, sample_names = val_batch
                    y_val_np = y_cpu.numpy() if torch.is_tensor(y_cpu) else np.asarray(y_cpu)
                    if torch.is_tensor(mask_cpu):
                        mask_val_np = mask_cpu.numpy().astype(np.float32)
                    else:
                        mask_val_np = np.asarray(mask_cpu, dtype=np.float32)

                    x_val = _to_input_tensor(x_cpu, device)
                    b_val = _to_brainiac_tensor(b_cpu, device) if use_brainiac_fusion else None
                    _, y_val_onehot, mask_val = _to_target_tensors(y_cpu, mask_cpu, device)
                    _validate_batch_inputs(
                        x_val, b_val, mask_val, sample_names, context="validation"
                    )

                    with _autocast_context(device, val_amp_enabled, amp_dtype):
                        val_logits, loss_val, loss_per_class_val = _compute_model_loss(
                            validation_model,
                            x_val,
                            b_val,
                            y_val_onehot,
                            mask_val,
                            cw_t,
                            alpha_t,
                            beta_t,
                            tversky_gamma,
                            wce_weight,
                            label_smoothing,
                            use_brainiac_fusion,
                            False,
                            0.0,
                        )

                    val_outputs_finite = (
                        torch.isfinite(val_logits).all().item()
                        and torch.isfinite(loss_val).all().item()
                        and torch.isfinite(loss_per_class_val).all().item()
                    )
                    if not val_outputs_finite and val_amp_enabled:
                        val_amp_fallback_count += 1
                        print(
                            f"WARNING: Non-finite {amp_dtype_name} validation output; "
                            f"retrying in FP32; samples=[{_sample_names_text(sample_names)}]"
                        )
                        with _autocast_context(device, False, None):
                            val_logits, loss_val, loss_per_class_val = _compute_model_loss(
                                validation_model,
                                x_val,
                                b_val,
                                y_val_onehot,
                                mask_val,
                                cw_t,
                                alpha_t,
                                beta_t,
                                tversky_gamma,
                                wce_weight,
                                label_smoothing,
                                use_brainiac_fusion,
                                False,
                                0.0,
                            )
                        val_outputs_finite = (
                            torch.isfinite(val_logits).all().item()
                            and torch.isfinite(loss_val).all().item()
                            and torch.isfinite(loss_per_class_val).all().item()
                        )
                        if val_outputs_finite and val_amp_fallback_count >= 3:
                            val_amp_enabled = False
                            print(
                                "WARNING: Disabling validation autocast after "
                                f"{val_amp_fallback_count} successful FP32 fallbacks."
                            )
                    if not val_outputs_finite:
                        raise FloatingPointError(
                            "Non-finite validation output after FP32 retry; "
                            f"loss={loss_val.detach().float().item()}, "
                            f"logits={_tensor_range(val_logits)}, input={_tensor_range(x_val)}, "
                            f"samples=[{_sample_names_text(sample_names)}]"
                        )
                    _update_loss_metrics(val_acc, loss_val.detach().float().cpu(), loss_per_class_val.detach().float())
                    val_probabilities = _probabilities_for_metrics(val_logits)
                    _update_prediction_metrics(val_acc, val_probabilities, y_val_np, mask_val_np)

            del val_loader
            _cleanup_temp_dirs(val_temp_dirs)
            gc.collect()

            val_s = time.perf_counter() - t0_val
            val_class_metrics, val_all_classes_metrics = _metrics_for_logging(val_acc)
            append_metrics_to_file(
                val_metrics_file_path,
                epoch + 1,
                val_class_metrics,
                all_classes_metrics=val_all_classes_metrics,
            )
            print(f"\nEpoch {epoch+1} Validation Report:")
            for class_index, metrics in val_class_metrics.items():
                print(
                    f"Class {class_index} - IoU: {metrics['iou']:.3f}, Acc: {metrics['accuracy']:.3f}, "
                    f"Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}, "
                    f"Loss: {metrics['loss']:.4f}"
                )
            print(
                "ALL_CLASSES (VAL) - MicroAcc: {0:.3f}, W-Prec: {1:.3f}, W-Rec: {2:.3f}, Loss: {3:.3f}".format(
                    val_all_classes_metrics["accuracy"],
                    val_all_classes_metrics["precision"],
                    val_all_classes_metrics["recall"],
                    val_all_classes_metrics["loss"],
                )
            )

        append_training_stats(
            training_stats_file_path,
            epoch=epoch + 1,
            data_load_s=data_load_s,
            train_s=train_s,
            val_s=val_s,
            slices_per_sec=slices_per_sec,
            mean_grad_norm=mean_grad_norm,
            learning_rate=learning_rate,
            vram_used_mb=vram_used_mb,
            vram_peak_mb=vram_peak_mb,
            dataloader_wait_s=mean_dataloader_wait_s,
            batch_compute_s=mean_batch_compute_s,
        )

    # End-of-training cleanup
    if train_loader is not None:
        del train_loader
    _cleanup_training_temp_cache()
    atexit.unregister(_cleanup_training_temp_cache)
    _worker_pool.shutdown(wait=False)
    print("Training completed.")
    log_file.close()


if __name__ == "__main__":
    train_model()
