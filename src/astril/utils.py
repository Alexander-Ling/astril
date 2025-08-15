import sys
import os
import json
import math
import numpy as np
import re
import ast
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
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
        "beta_vals_list": beta_vals_list
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
    # 1) Flatten or keep shape? We'll mask out invalid pixels by multiplying the loss by 'mask_f'
    boolean_mask = tf.cast(mask[..., 0] > 0.5, tf.float32)  # (batch, H, W, out_slices)
    mask_4d = tf.expand_dims(boolean_mask, axis=-1)         # (batch, H, W, out_slices, 1)

    # 2) Optionally check if y_pred is logits or already prob
    #    Typically crossentropy expects logits. If your net outputs prob, we use a stable trick:
    #    loss = -y_true * log(y_pred)  (no tf.nn.sparse_softmax_cross_entropy_with_logits).
    #    We'll assume y_pred is LOGITS here, so we do a typical cross_entropy below.

    # 3) Weighted crossentropy
    #    shape: y_true => (..., num_classes)
    #           y_pred => (..., num_classes)
    #    We do: CE = - sum_i [ w_i * y_true[i] * log(softmax(y_pred)[i]) ]
    #    Then sum/mean over valid pixels.

    # Turn logits into prob via softmax
    y_pred_prob = tf.nn.softmax(y_pred, axis=-1)

    # Prevent log(0)
    y_pred_prob = tf.clip_by_value(y_pred_prob, smooth, 1.0 - smooth)

    ce_loss_map = 0.0
    for c, w_c in enumerate(class_weights):
        # y_true_c => (batch, H, W, out_slices)
        # y_pred_prob_c => same shape
        y_true_c = y_true[..., c]
        y_pred_prob_c = y_pred_prob[..., c]

        # Weighted cross entropy for class c
        ce_loss_c = - w_c * y_true_c * tf.math.log(y_pred_prob_c + smooth)

        ce_loss_map += ce_loss_c

    # Multiply by mask to ignore invalid pixels
    ce_loss_map = ce_loss_map * tf.squeeze(mask_4d, axis=-1)

    # Now average over all valid pixels
    valid_pixels = tf.reduce_sum(boolean_mask)
    ce_loss = tf.reduce_sum(ce_loss_map) / (valid_pixels + smooth)

    return ce_loss

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

    # 1) Convert logits -> probabilities
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # 2) Mask out invalid pixels
    boolean_mask = tf.cast(mask[..., 0] > 0.5, tf.float32)
    mask_expanded = tf.expand_dims(boolean_mask, axis=-1)

    y_true = y_true * mask_expanded
    y_pred = y_pred * mask_expanded

    # 3) Loop over each class
    num_classes = y_true.shape[-1]
    batch_loss_sum = 0.0

    for c in range(num_classes):
        w_c = class_weights[c]
        alpha_c = alpha_vals[c]
        beta_c = beta_vals[c]

        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]

        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum(y_pred_c * (1.0 - y_true_c))
        fn = tf.reduce_sum((1.0 - y_pred_c) * y_true_c)

        tversky_c = (intersection + smooth) / (intersection + alpha_c * fp + beta_c * fn + smooth)
        # Focal Tversky
        focal_tversky_c = tf.pow((1.0 - tversky_c), gamma)

        batch_loss_sum += w_c * focal_tversky_c

    # 4) Average across classes
    loss_value = batch_loss_sum / float(num_classes)
    return loss_value

def combined_focal_tversky_wce_loss(
    y_true,
    y_pred,
    mask,
    class_weights,
    alpha_vals,
    beta_vals,
    gamma=1.0,
    wce_weight=0.5,
    smooth=1e-6
):
    """
    Combined Weighted Cross Entropy + Focal Tversky Loss.
    Returns:
      total_loss: tf.Tensor scalar for the entire batch
      per_class_loss: list (or tf.Tensor) of length num_classes giving the
                      combined loss contribution for each class.
    """

    # ------------------------------------------------------
    # 1) Weighted Cross Entropy for each class
    # ------------------------------------------------------
    # Convert logits -> probabilities
    y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
    y_pred_prob = tf.clip_by_value(y_pred_prob, smooth, 1.0 - smooth)

    # boolean_mask => shape (batch,H,W,out_slices)
    boolean_mask = tf.cast(mask[..., 0] > 0.5, tf.float32)
    # We'll flatten or at least sum over valid pixels
    valid_pixels = tf.reduce_sum(boolean_mask) + smooth

    # For partial loss tracking
    num_classes = y_true.shape[-1]

    per_class_wce_sum = []
    for c in range(num_classes):
        w_c = class_weights[c]

        y_true_c = y_true[..., c]  # shape (batch,H,W,out_slices)
        y_pred_c = y_pred_prob[..., c]

        # cross-entropy = - w_c * y_true_c * log(y_pred_c)
        ce_c = -w_c * y_true_c * tf.math.log(y_pred_c + smooth)
        # zero out invalid pixels
        ce_c = ce_c * boolean_mask

        # sum over the whole batch
        class_ce_sum = tf.reduce_sum(ce_c)
        per_class_wce_sum.append(class_ce_sum)

    # Weighted CE for the entire batch
    total_wce = tf.add_n(per_class_wce_sum) / valid_pixels

    # ------------------------------------------------------
    # 2) Focal Tversky for each class
    # ------------------------------------------------------
    y_pred_prob_masked = y_pred_prob * tf.expand_dims(boolean_mask, axis=-1)
    y_true_masked = y_true * tf.expand_dims(boolean_mask, axis=-1)

    per_class_ft = []
    for c in range(num_classes):
        w_c       = class_weights[c]
        alpha_c   = alpha_vals[c]
        beta_c    = beta_vals[c]

        y_true_c  = y_true_masked[..., c]
        y_pred_c  = y_pred_prob_masked[..., c]

        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum(y_pred_c * (1.0 - y_true_c))
        fn = tf.reduce_sum((1.0 - y_pred_c) * y_true_c)

        tversky_c = (intersection + smooth) / (intersection + alpha_c*fp + beta_c*fn + smooth)
        focal_tversky_c = tf.pow((1.0 - tversky_c), gamma)

        # weight the class’s focal tversky
        per_class_ft.append(w_c * focal_tversky_c)

    # Divide the Tversky Sum by the Sum of the Weights
    sum_w = tf.reduce_sum(class_weights)
    total_ft = tf.add_n(per_class_ft) / (sum_w + 1e-6)

    # ------------------------------------------------------
    # 3) Combine them
    # ------------------------------------------------------
    total_loss = wce_weight * total_wce + (1.0 - wce_weight) * total_ft

    # For each class c, define that class’s portion:
    #   class_loss_c = wce_weight*(per_class_wce_sum[c]/valid_pixels)
    #                  + (1-wce_weight)*(per_class_ft[c]/num_classes)
    per_class_loss = []
    for c in range(num_classes):
        class_wce = per_class_wce_sum[c] / valid_pixels
        class_ft  = per_class_ft[c] / (sum_w + 1e-6)
        per_class_loss_c = wce_weight * class_wce + (1.0 - wce_weight) * class_ft
        per_class_loss.append(per_class_loss_c)

    return total_loss, tf.stack(per_class_loss, axis=0)  # shape (num_classes,)

# -----------------------------------------------------------------------------
# Training/Validation performance loggers
# -----------------------------------------------------------------------------

def append_metrics_to_file(file_path, epoch, class_metrics, all_classes_metrics=None):
    """
    Append metrics for each class to a .tsv or .txt file.
    Optionally also write a row for "All_Classes" if all_classes_metrics is provided.
    """
    mode = 'a' if os.path.exists(file_path) else 'w'
    headers = "Epoch\tClass\tAccuracy\tPrecision\tRecall\tLoss\n"
    
    with open(file_path, mode) as file:
        if mode == 'w':
            file.write(headers)

        # Per-class lines
        for class_index, metrics in class_metrics.items():
            accuracy  = metrics['accuracy'].result().numpy()
            precision = metrics['precision'].result().numpy()
            recall    = metrics['recall'].result().numpy()
            loss_val  = metrics['loss'].result().numpy()
            file.write(
                f"{epoch}\tClass_{class_index}\t{accuracy:.3f}\t{precision:.3f}\t{recall:.3f}\t{loss_val:.4f}\n"
            )

        # Optional line for "All_Classes"
        if all_classes_metrics is not None:
            file.write(
                f"{epoch}\tAll_Classes\t"
                f"{all_classes_metrics['accuracy']:.3f}\t"
                f"{all_classes_metrics['precision']:.3f}\t"
                f"{all_classes_metrics['recall']:.3f}\t"
                f"{all_classes_metrics['loss']:.4f}\n"
            )

def prepare_class_metrics_for_logging(train_or_val_class_metrics):
    """
    Prepare a dictionary of metrics for each class index.
    """
    class_metrics = {}
    num_cls = len(train_or_val_class_metrics['accuracy'])
    for class_index in range(num_cls):
        class_metrics[class_index] = {
            'accuracy':  train_or_val_class_metrics['accuracy'][class_index],
            'precision': train_or_val_class_metrics['precision'][class_index],
            'recall':    train_or_val_class_metrics['recall'][class_index],
            'loss':      train_or_val_class_metrics['loss'][class_index],
        }
    return class_metrics

# -----------------------------------------------------------------------------
# Training Helpers (Checkpointing + Schedules)
# -----------------------------------------------------------------------------
def get_latest_checkpoint(checkpoint_dir):
    """
    Returns the latest checkpoint (either a directory named "epoch_{num}" 
    or a single-file .keras checkpoint named "epoch_{num}.keras") in 
    `checkpoint_dir`, based on modification time.

    If no checkpoint is found, returns None.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    all_items = os.listdir(checkpoint_dir)
    # Patterns to match single-file checkpoints or directories:
    keras_pattern = re.compile(r'^epoch_(\d+)\.keras$')
    dir_pattern = re.compile(r'^epoch_(\d+)$')

    valid_items = []
    for item in all_items:
        item_path = os.path.join(checkpoint_dir, item)
        
        # Check if it's a .keras file with the right pattern
        if os.path.isfile(item_path) and keras_pattern.match(item):
            valid_items.append(item_path)
        # Or if it's a directory with the right pattern
        elif os.path.isdir(item_path) and dir_pattern.match(item):
            valid_items.append(item_path)

    if not valid_items:
        return None

    # Return the item with the latest modification time
    latest_checkpoint = max(valid_items, key=os.path.getmtime)
    return latest_checkpoint

def get_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts the epoch number from a checkpoint path that may be 
    either a .keras file ("epoch_{number}.keras") or a directory 
    ("epoch_{number}").

    If neither pattern is matched, returns None.
    """
    filename = os.path.basename(checkpoint_path)

    # We allow either a .keras file or a directory
    patterns = [
        r'^epoch_(\d+)\.keras$',  # e.g., epoch_3.keras
        r'^epoch_(\d+)$'         # e.g., epoch_3 (directory format)
    ]

    for pat in patterns:
        match = re.search(pat, filename)
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
