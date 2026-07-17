import sys
import os
import json
import math
import numpy as np
import re
import ast
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime


# -----------------------------------------------------------------------------
# General Logging + File I/O
# -----------------------------------------------------------------------------
class FlushFile:
    """
    File-like wrapper that flushes the stream after each write.
    Useful to ensure logs are written immediately.
    """
    def __init__(self, f):
        self.f = f
    
    def write(self, x):
        self.f.write(x)
        self.f.flush()
    
    def flush(self):
        self.f.flush()


def init_logging(output_dir):
    """
    Initialize logging by redirecting stdout and stderr 
    to a timestamped log file within output_dir.

    Returns:
        (log_file, log_file_path)
    """
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(output_dir, f"Log_{current_date_time}.log")

    # Open the log file in buffered mode
    log_file = open(log_file_path, "w", buffering=1)

    # Replace stdout and stderr
    sys.stdout = FlushFile(log_file)
    sys.stderr = FlushFile(log_file)

    return log_file, log_file_path


def save_indexes(indexes, file_path):
    """
    Save array of indexes to a JSON file (for reproducible training).
    """
    with open(file_path, 'w') as f:
        json.dump(indexes.tolist(), f)  # Convert numpy array to list


def load_indexes(file_path):
    """
    Load array of indexes from a JSON file.
    """
    with open(file_path, 'r') as f:
        return np.array(json.load(f))

# -----------------------------------------------------------------------------
# Training Schedule: Function to parse and validate training schedule
# -----------------------------------------------------------------------------

# A small helper that raises an ERROR if a parameter that must not have a default is invalid
def get_required_value(current_params, key, parse_func, condition=None, err_msg="Invalid value"):
    """
    parse_func is something like int or float or a custom converter.
    condition is a lambda that checks validity (e.g. lambda x: x>0).
    If invalid, raise ValueError (error).
    """
    if key not in current_params:
        raise ValueError(f"Missing required parameter '{key}'. {err_msg}")
    raw_val = current_params[key]

    # Try converting
    try:
        val = parse_func(raw_val)
    except Exception:
        raise ValueError(f"Error parsing '{key}': {raw_val}. {err_msg}")

    # Check condition
    if condition and not condition(val):
        raise ValueError(f"Parameter '{key}' has invalid value: {raw_val}. {err_msg}")

    return val

# A small helper that issues a WARNING if invalid, then returns a default
def get_optional_value(current_params, key, parse_func, default, warn_msg, condition=None):
    """
    parse_func is something like int or float or a custom converter.
    If invalid or missing, produce warning & use 'default'.
    Otherwise return parsed value.
    """
    raw_val = current_params.get(key, None)
    if raw_val is None:
        # param not provided => warning
        print(f"WARNING: Parameter '{key}' not provided. {warn_msg}. Using default={default}")
        return default

    # Attempt to parse
    try:
        val = parse_func(raw_val)
    except Exception:
        print(f"WARNING: Parameter '{key}' invalid: {raw_val}. {warn_msg}. Using default={default}")
        return default

    # Check condition
    if condition and not condition(val):
        print(f"WARNING: Parameter '{key}' out of valid range: {raw_val}. {warn_msg}. Using default={default}")
        return default

    return val

# Function to parse dictionary sring
def parse_dict_string(dict_str):
    """
    Safely parse a dictionary-like string (e.g., "{(1,2,3):5, (1,3):2}")
    into a Python dictionary.

    If dict_str is empty, 'nan', invalid, or 'None', returns empty dict.
    """
    if not dict_str or dict_str in ["{}", "None", "null", "nan"]:
        return {}
    try:
        parsed = ast.literal_eval(dict_str)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed object is not a dictionary.")
        for k in parsed.keys():
            if not isinstance(k, tuple):
                raise ValueError(f"Invalid key {k}. Keys must be tuples.")
        return parsed
    except Exception as e:
        raise ValueError(f"Error parsing dictionary string '{dict_str}': {e}")

# Deal with NA strings to prevent them being turned into floats
def parse_str_or_none(raw_value):
    """
    Convert raw_value to a lowercased string (if possible),
    then return None if it's in [ 'na', 'nan', 'none', '{}', '' ].
    Otherwise, return the stripped lower string.
    """
    if raw_value is None:
        return None
    s = str(raw_value).strip().lower()
    if s in ["na", "nan", "none", "{}", ""]:
        return None
    return s

# Main training schedule parsing function
def parse_and_validate_schedule_params(
    current_params: dict,
    num_classes: int,
    train_indexes: np.ndarray
):
    """
    Parse & validate a schedule row (current_params) for the current epoch.
    Returns a dict containing all validated parameters.

    Required parameters (no defaults -> error if invalid):
      - slice_sub_batch_size (int > 0)
      - accumulate_n_sub_batches (int > 0)
      - conduct_validation (bool -> True or False)
      - validation_frequency (int > 0) [only if conduct_validation is True]
      - learning_rate (float)

    Optional parameters (warnings if invalid -> fallback):
      - wce_weight (float in [0,1]) -> default=0.5
      - tversky_gamma (float > 0) -> default=1
      - class_weights (comma-separated -> length == num_classes) -> default=None
      - epochs_per_new_training_data (int > 0) -> default=1
      - scan_batch_size (int > 0) -> default=len(train_indexes)
      - class_multiplication_factors (dict-like) -> default={}
      - require_classes (dict-like) -> default={}
      - tversky_alpha_values (comma-separated -> length==num_classes in [0,1]) -> default=all 0.5
    """

    # ------------------ 1) Required parameters ------------------
    slice_sub_batch_size = get_required_value(
        current_params,
        "slice_sub_batch_size",
        int,
        condition=lambda x: x > 0,
        err_msg="Must be an integer > 0."
    )

    accumulate_n_sub_batches = get_required_value(
        current_params,
        "accumulate_n_sub_batches",
        int,
        condition=lambda x: x > 0,
        err_msg="Must be an integer > 0."
    )

    # We'll parse booleans via a small function:
    def parse_bool(raw):
        # e.g. 'TRUE' or 'False' or actual bool
        if isinstance(raw, bool):
            return raw
        low = str(raw).lower()
        if low in ['true', 't', '1']:
            return True
        elif low in ['false', 'f', '0']:
            return False
        else:
            raise ValueError("Not a valid boolean string.")

    conduct_validation = get_required_value(
        current_params,
        "conduct_validation",
        parse_bool,
        err_msg="Must be TRUE or FALSE."
    )

    validation_frequency = 1
    if conduct_validation:
        validation_frequency = get_required_value(
            current_params,
            "validation_frequency",
            int,
            condition=lambda x: x > 0,
            err_msg="Must be an integer > 0."
        )

    learning_rate = get_required_value(
        current_params,
        "learning_rate",
        float,
        condition=lambda x: x > 0,
        err_msg="Must be a float > 0."
    )

    # ------------------ 2) Optional parameters ------------------
    # wce_weight in [0,1], default=0.5
    wce_weight = get_optional_value(
        current_params,
        "wce_loss_weight",
        float,
        default=0.5,
        warn_msg="Must be a float in [0,1]",
        condition=lambda x: 0 <= x <= 1
    )

    # tversky_gamma > 0, default=1
    tversky_gamma = get_optional_value(
        current_params,
        "tversky_gamma",
        float,
        default=1.0,
        warn_msg="Must be a float > 0",
        condition=lambda x: x > 0
    )

    # 2a) class_weights => comma-separated or NA => default=None
    raw_class_weights = parse_str_or_none(current_params.get("class_weights", None))
    if raw_class_weights is None:
        class_weights = None
    else:
        split_str = raw_class_weights.split(',')
        if len(split_str) != num_classes:
            print(
                f"'class_weights' length={len(split_str)} != num_classes={num_classes}. "
                "Using default=None."
            )
            class_weights = None
        else:
            tmp_cw = []
            for i, w in enumerate(split_str):
                try:
                    fw = float(w.strip())
                    tmp_cw.append(fw)
                except Exception:
                    print(
                        f"WARNING: Invalid value in 'class_weights' at index {i}: {w}. "
                        "Using NA for this class."
                    )
                    # Instead of default=1.0, store NaN
                    tmp_cw.append(float('nan'))
            class_weights = tmp_cw

    # 2b) epochs_per_new_training_data => int>0 => default=1
    epochs_per_data = get_optional_value(
        current_params,
        "epochs_per_new_training_data",
        int,
        default=1,
        warn_msg="Must be integer > 0",
        condition=lambda x: x > 0
    )

    # 2c) scan_batch_size => int>0 => default=len(train_indexes)
    default_scan_bs = len(train_indexes) if len(train_indexes) > 0 else 1
    scan_batch_size = get_optional_value(
        current_params,
        "scan_batch_size",
        int,
        default=default_scan_bs,
        warn_msg=f"Must be integer > 0. Using default={default_scan_bs}",
        condition=lambda x: x > 0
    )
    if scan_batch_size > len(train_indexes) and len(train_indexes) > 0:
        print(
            f"WARNING:"
            f"scan_batch_size={scan_batch_size} > #train_samples={len(train_indexes)}. "
            f"Clamping to {len(train_indexes)}."
        )
        scan_batch_size = len(train_indexes)

    # 2d) class_multiplication_factors => dict-like or NA => default=None
    raw_cmf = parse_str_or_none(current_params.get("class_multiplication_factors", None))
    if raw_cmf is None:
        class_multiplication_factors = {}
    else:
        try:
            class_multiplication_factors = parse_dict_string(raw_cmf)
        except Exception:
            print(
                f"WARNING: Failed to parse 'class_multiplication_factors': {raw_cmf}."
            )
            class_multiplication_factors = {}

    # 2e) require_classes => dict-like or NA => default={}
    raw_rc = parse_str_or_none(current_params.get("require_classes", None))
    if raw_rc is None:
        require_classes = {}
    else:
        try:
            require_classes = parse_dict_string(raw_rc)
        except Exception:
            print(
                f"WARNING: Failed to parse 'require_classes': {raw_rc}."
            )
            require_classes = {}

    # 2f) tversky_alpha_values => comma-separated or NA => default=all 0.5
    alpha_vals_list = [0.5]*num_classes
    alpha_str = parse_str_or_none(current_params.get("tversky_alpha_values", None))
    if alpha_str is not None:
        # parse
        alpha_tokens = alpha_str.split(',')
        if len(alpha_tokens) != num_classes:
            print(
                f"WARNING:"
                f"Incorrect number of Tversky alpha values. "
                f"Expected={num_classes}, got={len(alpha_tokens)}. "
                "Using default=0.5 for all classes."
            )
        else:
            parsed_alphas = []
            for i, tok in enumerate(alpha_tokens):
                try:
                    valf = float(tok.strip())
                    if 0 <= valf <= 1:
                        parsed_alphas.append(valf)
                    else:
                        print(
                            f"WARNING: Alpha value {valf} out of [0,1]. Using 0.5 for class {i}."
                        )
                        parsed_alphas.append(0.5)
                except Exception:
                    print(
                        f"WARNING: Failed to parse alpha '{tok}' for class {i}. Using 0.5."
                    )
                    parsed_alphas.append(0.5)
            alpha_vals_list = parsed_alphas

    # compute beta for Tversky => (1 - alpha)
    beta_vals_list = [1.0 - a for a in alpha_vals_list]

    # 2g) gradient_clip_norm => float > 0 or NA => default=None (no clipping)
    gradient_clip_norm = get_optional_value(
        current_params,
        "gradient_clip_norm",
        float,
        default=None,
        warn_msg="Must be a float > 0",
        condition=lambda x: x > 0
    )

    # 2h) label_smoothing => float in [0, 0.5] => default=0.0
    label_smoothing = get_optional_value(
        current_params,
        "label_smoothing",
        float,
        default=0.0,
        warn_msg="Must be a float in [0, 0.5]",
        condition=lambda x: 0.0 <= x <= 0.5
    )

    # 2i) deep_supervision_loss_weight => float in [0, 1] => default=0.5
    deep_supervision_loss_weight = get_optional_value(
        current_params,
        "deep_supervision_loss_weight",
        float,
        default=0.5,
        warn_msg="Must be a float in [0, 1]",
        condition=lambda x: 0.0 <= x <= 1.0
    )

    return {
        "scan_batch_size": scan_batch_size,
        "slice_sub_batch_size": slice_sub_batch_size,
        "accumulate_n_sub_batches": accumulate_n_sub_batches,
        "conduct_validation": conduct_validation,
        "validation_frequency": validation_frequency,
        "learning_rate": learning_rate,
        "wce_weight": wce_weight,
        "tversky_gamma": tversky_gamma,
        "class_weights": class_weights,
        "epochs_per_data": epochs_per_data,
        "class_multiplication_factors": class_multiplication_factors,
        "require_classes": require_classes,
        "alpha_vals_list": alpha_vals_list,
        "beta_vals_list": beta_vals_list,
        "gradient_clip_norm": gradient_clip_norm,
        "label_smoothing": label_smoothing,
        "deep_supervision_loss_weight": deep_supervision_loss_weight,
    }

# -----------------------------------------------------------------------------
# Metrics: Weighted Focal Tversky Loss + Weighted Cross Entropy Loss
# -----------------------------------------------------------------------------
def weighted_cross_entropy(
    y_true,         # (batch, H, W, out_slices, num_classes) one-hot
    y_pred,         # same shape, typically logits or pre-softmax
    mask,           # (batch, H, W, out_slices, 1) to ignore invalid pixels
    class_weights,
    smooth=1e-6
):
    """
    Multi-class Weighted Cross Entropy, ignoring pixels outside 'mask'.
    We expect y_pred to be raw logits (not yet softmaxed), but if they're
    already softmax, you'll see a difference in usage below.
    """
    boolean_mask = (mask[..., 0] > 0.5).to(dtype=y_pred.dtype)
    y_pred_prob = F.softmax(y_pred, dim=-1).clamp(smooth, 1.0 - smooth)
    class_weights = torch.as_tensor(class_weights, dtype=y_pred.dtype, device=y_pred.device)
    ce_loss_map = -(class_weights * y_true * torch.log(y_pred_prob + smooth)).sum(dim=-1)
    ce_loss_map = ce_loss_map * boolean_mask
    return ce_loss_map.sum() / (boolean_mask.sum() + smooth)

def focal_tversky_loss(
    y_true,
    y_pred,
    mask,
    class_weights,       # from load_epoch_data
    alpha_vals,          # from schedule
    beta_vals,           # from schedule
    gamma=1.0,
    smooth=1e-6
):
    """
    Multi-class Focal Tversky loss with:
      - class_weights per class
      - alpha_vals, beta_vals per class
      - optional focal exponent gamma
      - mask to ignore irrelevant pixels
    """

    y_pred = F.softmax(y_pred, dim=-1)
    boolean_mask = (mask[..., 0] > 0.5).to(dtype=y_pred.dtype)
    mask_expanded = boolean_mask.unsqueeze(-1)
    y_true = y_true * mask_expanded
    y_pred = y_pred * mask_expanded

    class_weights = torch.as_tensor(class_weights, dtype=y_pred.dtype, device=y_pred.device)
    alpha_vals = torch.as_tensor(alpha_vals, dtype=y_pred.dtype, device=y_pred.device)
    beta_vals = torch.as_tensor(beta_vals, dtype=y_pred.dtype, device=y_pred.device)

    per_class = []
    for c in range(y_true.shape[-1]):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        intersection = torch.sum(y_true_c * y_pred_c)
        fp = torch.sum(y_pred_c * (1.0 - y_true_c))
        fn = torch.sum((1.0 - y_pred_c) * y_true_c)
        tversky_c = (intersection + smooth) / (
            intersection + alpha_vals[c] * fp + beta_vals[c] * fn + smooth
        )
        per_class.append(class_weights[c] * torch.pow(1.0 - tversky_c, gamma))

    return torch.stack(per_class).sum() / float(y_true.shape[-1])

def combined_focal_tversky_wce_loss(
    y_true,
    y_pred,
    mask,
    class_weights,
    alpha_vals,
    beta_vals,
    gamma=1.0,
    wce_weight=0.5,
    smooth=1e-6,
    label_smoothing=0.0,
):
    """
    Combined Weighted Cross Entropy + Focal Tversky Loss.
    Returns:
      total_loss: torch.Tensor scalar for the entire batch
      per_class_loss: torch.Tensor of length num_classes giving the
                      combined loss contribution for each class.
    label_smoothing: if > 0, soft-labels the one-hot targets for the WCE component,
                     reducing overconfidence on rare classes.
    """

    # ------------------------------------------------------
    # 1) Weighted Cross Entropy for each class
    # ------------------------------------------------------
    # Apply label smoothing to one-hot targets before WCE (Tversky uses original targets)
    # The reductions below can span millions of pixels. Keep the loss in FP32
    # even when the model forward pass is running under CUDA autocast.
    y_pred = y_pred.float()
    y_true = y_true.float()
    mask = mask.float()
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=y_pred.device)
    alpha_vals = torch.as_tensor(alpha_vals, dtype=torch.float32, device=y_pred.device)
    beta_vals = torch.as_tensor(beta_vals, dtype=torch.float32, device=y_pred.device)

    if label_smoothing > 0.0:
        num_classes_ls = float(y_true.shape[-1])
        y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes_ls
    else:
        y_true_smooth = y_true

    y_pred_prob = F.softmax(y_pred, dim=-1).clamp(smooth, 1.0 - smooth)

    boolean_mask = (mask[..., 0] > 0.5).to(dtype=y_pred.dtype)
    valid_pixels = torch.sum(boolean_mask) + smooth
    num_classes = y_true.shape[-1]

    per_class_wce_sum = []
    for c in range(num_classes):
        w_c = class_weights[c]
        y_true_c = y_true_smooth[..., c]
        y_pred_c = y_pred_prob[..., c]
        ce_c = -w_c * y_true_c * torch.log(y_pred_c + smooth)
        ce_c = ce_c * boolean_mask
        class_ce_sum = torch.sum(ce_c)
        per_class_wce_sum.append(class_ce_sum)

    total_wce = torch.stack(per_class_wce_sum).sum() / valid_pixels

    y_pred_prob_masked = y_pred_prob * boolean_mask.unsqueeze(-1)
    y_true_masked = y_true * boolean_mask.unsqueeze(-1)

    per_class_ft = []
    for c in range(num_classes):
        w_c = class_weights[c]
        alpha_c = alpha_vals[c]
        beta_c = beta_vals[c]
        y_true_c = y_true_masked[..., c]
        y_pred_c = y_pred_prob_masked[..., c]

        intersection = torch.sum(y_true_c * y_pred_c)
        fp = torch.sum(y_pred_c * (1.0 - y_true_c))
        fn = torch.sum((1.0 - y_pred_c) * y_true_c)

        tversky_c = (intersection + smooth) / (intersection + alpha_c*fp + beta_c*fn + smooth)
        focal_tversky_c = torch.pow((1.0 - tversky_c), gamma)
        per_class_ft.append(w_c * focal_tversky_c)

    sum_w = torch.sum(class_weights)
    total_ft = torch.stack(per_class_ft).sum() / (sum_w + 1e-6)

    total_loss = wce_weight * total_wce + (1.0 - wce_weight) * total_ft

    per_class_loss = []
    for c in range(num_classes):
        class_wce = per_class_wce_sum[c] / valid_pixels
        class_ft  = per_class_ft[c] / (sum_w + 1e-6)
        per_class_loss_c = wce_weight * class_wce + (1.0 - wce_weight) * class_ft
        per_class_loss.append(per_class_loss_c)

    return total_loss, torch.stack(per_class_loss, dim=0)

# -----------------------------------------------------------------------------
# Training/Validation performance loggers
# -----------------------------------------------------------------------------

def append_metrics_to_file(file_path, epoch, class_metrics, all_classes_metrics=None):
    """
    Append metrics for each class to a .tsv or .txt file.
    Optionally also write a row for "All_Classes" if all_classes_metrics is provided.
    """
    legacy_header = "Epoch\tClass\tAccuracy\tPrecision\tRecall\tLoss"
    headers = "Epoch\tClass\tIoU\tAccuracy\tPrecision\tRecall\tLoss\n"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as existing_file:
            existing_lines = existing_file.readlines()
        if existing_lines and existing_lines[0].rstrip("\r\n") == legacy_header:
            upgraded_lines = [headers]
            for line in existing_lines[1:]:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 6:
                    raise ValueError(f"Cannot upgrade malformed legacy metrics row in {file_path}: {line!r}")
                epoch_value, class_name, legacy_accuracy, precision, recall, loss = fields
                # Historical per-class "Accuracy" was IoU. The historical
                # All_Classes value was micro accuracy, so it has no matching IoU.
                if class_name == "All_Classes":
                    iou, accuracy = "NA", legacy_accuracy
                else:
                    iou, accuracy = legacy_accuracy, "NA"
                upgraded_lines.append(
                    f"{epoch_value}\t{class_name}\t{iou}\t{accuracy}\t{precision}\t{recall}\t{loss}\n"
                )
            with open(file_path, "w", encoding="utf-8", newline="") as upgraded_file:
                upgraded_file.writelines(upgraded_lines)

    mode = 'a' if os.path.exists(file_path) else 'w'
    
    with open(file_path, mode) as file:
        if mode == 'w':
            file.write(headers)

        # Per-class lines
        for class_index, metrics in class_metrics.items():
            iou = _metric_value(metrics['iou'])
            accuracy = _metric_value(metrics['accuracy'])
            precision = _metric_value(metrics['precision'])
            recall = _metric_value(metrics['recall'])
            loss_val = _metric_value(metrics['loss'])
            file.write(
                f"{epoch}\tClass_{class_index}\t{iou:.3f}\t{accuracy:.3f}\t{precision:.3f}\t{recall:.3f}\t{loss_val:.4f}\n"
            )

        # Optional line for "All_Classes"
        if all_classes_metrics is not None:
            file.write(
                f"{epoch}\tAll_Classes\tNA\t"
                f"{all_classes_metrics['accuracy']:.3f}\t"
                f"{all_classes_metrics['precision']:.3f}\t"
                f"{all_classes_metrics['recall']:.3f}\t"
                f"{all_classes_metrics['loss']:.4f}\n"
            )


def _metric_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)

# -----------------------------------------------------------------------------
# Training Helpers (Checkpointing + Schedules)
# -----------------------------------------------------------------------------
def get_latest_checkpoint(checkpoint_dir):
    """
    Returns the latest PyTorch checkpoint named "epoch_{num}.pt" in
    `checkpoint_dir`, based on modification time.

    If no checkpoint is found, returns None.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    all_items = os.listdir(checkpoint_dir)
    pt_pattern = re.compile(r'^epoch_(\d+)\.pt$')

    valid_items = []
    for item in all_items:
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isfile(item_path) and pt_pattern.match(item):
            valid_items.append(item_path)

    if not valid_items:
        return None

    # Return the item with the latest modification time
    latest_checkpoint = max(valid_items, key=os.path.getmtime)
    return latest_checkpoint

def get_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts the epoch number from a PyTorch checkpoint path named
    "epoch_{number}.pt".

    If neither pattern is matched, returns None.
    """
    filename = os.path.basename(checkpoint_path)

    match = re.search(r'^epoch_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))

    return None

def get_parameters_for_epoch(epoch, training_schedule):
    """
    Get the most recent parameter row from the schedule that is <= the current epoch.
    """
    applicable_schedule = training_schedule[training_schedule['epoch'] <= epoch]
    return applicable_schedule.iloc[-1]

# -----------------------------------------------------------------------------
# compute_masked_predictions()
# -----------------------------------------------------------------------------
def compute_masked_predictions(probabilities, ground_truth, mask, num_classes):
    """
    Given:
      - probabilities: raw model output: 
         single-slice => (batch, H, W, classes)
         multi-slice   => (batch, H, W, out_slices, classes)
      - ground_truth: 
         single-slice => (batch, H, W, 1)
         multi-slice   => (batch, H, W, out_slices, 1)
      - mask: 
         single-slice => (batch, H, W, 1)
         multi-slice   => (batch, H, W, out_slices, 1)
      - num_classes: total number of segmentation classes

    Returns:
      masked_predictions_filtered, masked_ground_truth_filtered
      Both are flattened 1D arrays containing only ROI pixels.
    """

    # 1) Unify shapes so that we treat everything as (batch, H, W, out_slices)
    #    Then do argmax => shape (batch, H, W, out_slices).

    if probabilities.ndim == 4:
        # shape (batch, H, W, classes) => single-slice
        # we treat out_slices=1
        # Argmax => (batch, H, W)
        predictions = np.argmax(probabilities, axis=-1)  # shape => (batch, H, W)
        # Expand dims to (batch, H, W, 1) for consistency
        predictions = np.expand_dims(predictions, axis=-1)
    else:
        # shape => (batch, H, W, out_slices, classes)
        # Argmax => (batch, H, W, out_slices)
        predictions = np.argmax(probabilities, axis=-1)

    # ground_truth => might be (batch,H,W,1) or (batch,H,W,out_slices,1)
    if ground_truth.ndim == 4:
        # shape => (batch, H, W, 1)
        # expand => (batch, H, W, 1, 1) or we can expand to (batch,H,W,1)
        # We'll unify to (batch,H,W, out_slices=1)
        ground = np.squeeze(ground_truth, axis=-1)  # => (batch,H,W)
        ground = np.expand_dims(ground, axis=-1)    # => (batch,H,W,1)
    else:
        # shape => (batch,H,W,out_slices,1)
        # squeeze last dim => (batch,H,W,out_slices)
        ground = np.squeeze(ground_truth, axis=-1)  # => (batch,H,W,out_slices)

    # mask => might be (batch,H,W,1) or (batch,H,W,out_slices,1)
    if mask.ndim == 4:
        # shape => (batch,H,W,1)
        m = np.squeeze(mask, axis=-1)  # => (batch,H,W)
        m = np.expand_dims(m, axis=-1) # => (batch,H,W,1)
    else:
        # shape => (batch,H,W,out_slices,1)
        m = np.squeeze(mask, axis=-1)  # => (batch,H,W,out_slices)

    # Now predictions, ground, and m should all have shape:
    # (batch, H, W, out_slices)
    # If out_slices=1 => shape => (batch, H, W, 1)

    # 2) Apply mask => -1 outside ROI
    # np.where shapes must match
    if predictions.shape != m.shape:
        # Expand dims or squeeze to unify
        # If m has shape (batch,H,W) and predictions has (batch,H,W,1), or vice-versa, fix that
        if len(m.shape) == 3 and len(predictions.shape) == 4:
            # expand m
            m = np.expand_dims(m, axis=-1)
        elif len(m.shape) == 4 and len(predictions.shape) == 3:
            # expand predictions
            predictions = np.expand_dims(predictions, axis=-1)
            ground = np.expand_dims(ground, axis=-1)

    masked_predictions = np.where(m == 1, predictions, -1)
    masked_gt = np.where(m == 1, ground, -1)

    # 3) Flatten
    masked_predictions_flat = masked_predictions.flatten()
    masked_gt_flat = masked_gt.flatten()

    # 4) Filter out -1
    roi_indices = np.where(
        (masked_predictions_flat != -1) & (masked_gt_flat != -1)
    )[0]
    masked_predictions_filtered = masked_predictions_flat[roi_indices]
    masked_ground_truth_filtered = masked_gt_flat[roi_indices]

    return masked_predictions_filtered, masked_ground_truth_filtered


# ---------------------------------------------------------------------
# Function to calculate weighted macro metrics
# ---------------------------------------------------------------------

def compute_weighted_macro_metrics(agg, num_classes, epsilon=1e-9):
    """
    Given the aggregator dict containing:
      - correct_by_class[c]
      - gt_count_by_class[c]
      - pred_count_by_class[c]
      - total_samples
    compute Weighted Macro Precision/Recall and Micro (overall) Accuracy.
    Returns a dict: {
      'weighted_macro_precision': float,
      'weighted_macro_recall': float,
      'micro_accuracy': float
    }
    """
    correct_by_class = agg['correct_by_class']
    gt_count_by_class = agg['gt_count_by_class']
    pred_count_by_class = agg['pred_count_by_class']
    total_gt = gt_count_by_class.sum()

    # micro accuracy = total correct / total samples
    total_correct = correct_by_class.sum()
    micro_accuracy = total_correct / float(agg['total_samples'] + epsilon)

    if total_gt < 1:
        # Edge case: no ground truth? Return zeros
        return {
            'weighted_macro_precision': 0.0,
            'weighted_macro_recall': 0.0,
            'micro_accuracy': micro_accuracy
        }

    weighted_prec_sum = 0.0
    weighted_recall_sum = 0.0

    for c in range(num_classes):
        tp = correct_by_class[c]
        fp = pred_count_by_class[c] - tp
        fn = gt_count_by_class[c] - tp

        prec_c = tp / float(tp + fp + epsilon)
        recall_c = tp / float(tp + fn + epsilon)

        class_support = gt_count_by_class[c]
        w_c = class_support / float(total_gt)  # weight for class c

        weighted_prec_sum += w_c * prec_c
        weighted_recall_sum += w_c * recall_c

    return {
        'weighted_macro_precision': weighted_prec_sum,
        'weighted_macro_recall': weighted_recall_sum,
        'micro_accuracy': micro_accuracy
    }


# -----------------------------------------------------------------------------
# Hardware stats and training stats logging
# -----------------------------------------------------------------------------

def get_vram_stats_mb():
    """
    Returns (current_mb, peak_mb) GPU memory usage.
    Returns (0.0, 0.0) if no CUDA GPU is available.
    """
    try:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_allocated() / 1024 ** 2,
        )
    except Exception:
        return 0.0, 0.0


def append_training_stats(
    file_path,
    epoch,
    data_load_s,
    train_s,
    val_s,
    slices_per_sec,
    mean_grad_norm,
    learning_rate,
    vram_used_mb,
    vram_peak_mb,
    dataloader_wait_s=None,
    batch_compute_s=None,
):
    """
    Appends one row per epoch to training_stats.tsv.
    Writes header on first call (when file does not yet exist).
    """
    header = (
        "Epoch\tDataLoad_s\tTrain_s\tVal_s\t"
        "Slices_Per_Sec\tMean_Grad_Norm\tLR\tVRAM_Used_MB\tVRAM_Peak_MB\t"
        "Mean_DataLoader_Wait_s\tMean_Batch_Compute_s\n"
    )
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            lines = f.readlines()
        existing_header = lines[0].rstrip("\n")
        if "Mean_DataLoader_Wait_s" not in existing_header:
            upgraded = [
                existing_header + "\tMean_DataLoader_Wait_s\tMean_Batch_Compute_s\n"
            ]
            for line in lines[1:]:
                upgraded.append(line.rstrip("\n") + "\tNA\tNA\n")
            with open(file_path, "w") as f:
                f.writelines(upgraded)

    mode = 'a' if os.path.exists(file_path) else 'w'
    val_s_str = f"{val_s:.1f}" if val_s is not None else "NA"
    wait_s_str = f"{dataloader_wait_s:.4f}" if dataloader_wait_s is not None else "NA"
    compute_s_str = f"{batch_compute_s:.4f}" if batch_compute_s is not None else "NA"
    with open(file_path, mode) as f:
        if mode == 'w':
            f.write(header)
        f.write(
            f"{epoch}\t{data_load_s:.1f}\t{train_s:.1f}\t{val_s_str}\t"
            f"{slices_per_sec:.0f}\t{mean_grad_norm:.4f}\t{learning_rate:.6f}\t"
            f"{vram_used_mb:.0f}\t{vram_peak_mb:.0f}\t{wait_s_str}\t{compute_s_str}\n"
        )
