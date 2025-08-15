import os
import gc
import random
import math
import numpy as np
import psutil
import re
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Keras imports
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K

# Import configuration values.
from .config import (
    output_dir,
    image_paths_files,      # training channels config file list (comma?separated paths)
    gt_paths_file,          # training ground truth file
    mask_paths_file,        # training mask file
    val_image_paths_files,  # validation channels config file list
    val_gt_paths_file,      # validation ground truth file
    val_mask_paths_file,    # validation mask file
    num_classes,
    epochs,
    num_channels,
    slicing_plane,
    training_schedule_file,
    pretrained_model_path,
    print_every_n_subbatches,
    num_input_slices,
    num_output_slices,
    minimum_height_width,
    n_cores
)
from .data_loading import (
    read_paths_from_file,
    detect_input_shape,
    debug_plot_25d_slices,
    load_epoch_data,
    load_val_data,
    EpochDataGenerator,
    ValDataGenerator
)
from .model_architecture import (
    create_dynamic_unet_from_config,
    ResidualConvBlock,
    AttentionBlock,
    DynamicAttentionResUNet
)
from .utils import (
    init_logging,
    parse_and_validate_schedule_params,
    combined_focal_tversky_wce_loss,
    append_metrics_to_file,
    prepare_class_metrics_for_logging,
    get_latest_checkpoint,
    get_epoch_from_checkpoint,
    get_parameters_for_epoch,
    compute_masked_predictions,
    compute_weighted_macro_metrics
)

# Define custom objects for model loading.
custom_objects_dict = {
    'ResidualConvBlock': ResidualConvBlock,
    'AttentionBlock': AttentionBlock,
    'DynamicAttentionResUNet': DynamicAttentionResUNet
}

# Check TensorFlow version to decide checkpoint format.
_tf_version = tf.__version__.split('.')
tf_major = int(_tf_version[0])
tf_minor = int(_tf_version[1])
IS_NEWER_TF = (tf_major > 2) or (tf_major == 2 and tf_minor >= 9)

def get_checkpoint_name(epoch: int) -> str:
    if IS_NEWER_TF:
        return f"epoch_{epoch}.keras"
    else:
        return f"epoch_{epoch}"

# Create a global optimizer.
optimizer = tf.keras.optimizers.Adam()

def train_model():
    # 1. Initialize logging.
    log_file, log_file_path = init_logging(output_dir)
    print(f"Logging to: {log_file_path}")

    # 2. Load the training schedule from CSV/TSV.
    training_schedule = pd.read_csv(training_schedule_file, sep='\t')

    # 3. Load the training file lists.
    train_channel_paths = [read_paths_from_file(path) for path in image_paths_files]
    assert all(len(train_channel_paths[0]) == len(p) for p in train_channel_paths), \
        "Mismatch in the number of paths across training channels"
    train_volume_paths_list = list(zip(*train_channel_paths))
    train_volume_paths_list = [list(scan) for scan in train_volume_paths_list]
    train_mask_paths = read_paths_from_file(mask_paths_file)
    train_gt_paths = read_paths_from_file(gt_paths_file)

    # 4. Load the validation file lists.
    val_channel_paths = [read_paths_from_file(path) for path in val_image_paths_files]
    assert all(len(val_channel_paths[0]) == len(p) for p in val_channel_paths), \
        "Mismatch in the number of paths across validation channels"
    val_volume_paths_list = list(zip(*val_channel_paths))
    val_volume_paths_list = [list(scan) for scan in val_volume_paths_list]
    val_mask_paths = read_paths_from_file(val_mask_paths_file)
    val_gt_paths = read_paths_from_file(val_gt_paths_file)

    # 5. Detect a representative input shape for logging.
    input_shape, original_shape = detect_input_shape(
        sample_file_path=train_mask_paths[0],
        slicing_plane=slicing_plane,
        num_channels=num_channels
    )
    print(f"Original Shape: {original_shape}")
    print(f"Padded Input Shape: {input_shape}")

    # 6. Initialize or load model.
    model_subdir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_subdir, exist_ok=True)
    latest_checkpoint = get_latest_checkpoint(model_subdir)
    if latest_checkpoint:
        starting_epoch = get_epoch_from_checkpoint(latest_checkpoint)
        if starting_epoch is not None and starting_epoch >= epochs:
            print(f"Training completed up to epoch {starting_epoch}. Nothing more to do.")
            log_file.close()
            return
        if not IS_NEWER_TF and latest_checkpoint.endswith('.keras'):
            raise ValueError(
                f"TF version {tf.__version__} may not support .keras format checkpoint: {latest_checkpoint}"
            )
        print(f"Loading model from {latest_checkpoint}...")
        model = tf.keras.models.load_model(
            latest_checkpoint,
            custom_objects=custom_objects_dict
        )
    else:
        if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
            print(f"Starting from pre-trained model: {pretrained_model_path}")
            model = tf.keras.models.load_model(
                pretrained_model_path,
                custom_objects=custom_objects_dict
            )
            starting_epoch = 0
        else:
            print("Creating model from scratch.")
            model = create_dynamic_unet_from_config()
            starting_epoch = 0

    if latest_checkpoint is None and (pretrained_model_path is None or not os.path.exists(pretrained_model_path)):
        dummy_input_shape = (1, minimum_height_width, minimum_height_width, num_channels * num_input_slices)
        dummy_input = tf.zeros(dummy_input_shape, dtype=tf.float32)
        _ = model(dummy_input)
        print("Model built with dummy input shape:", dummy_input_shape)

    # 7. Define training and validation indexes as local variables.
    train_indexes = np.arange(len(train_mask_paths))
    val_indexes = np.arange(len(val_mask_paths))
    # (No need to save these to disk.)

    # 8. Create metric file paths.
    train_metrics_file_path = os.path.join(output_dir, "train_metrics.tsv")
    val_metrics_file_path = os.path.join(output_dir, "val_metrics.tsv")

    # 9. Initialize per-class and overall Keras metrics.
    train_class_metrics = {
        'accuracy':  [tf.keras.metrics.Accuracy() for _ in range(num_classes)],
        'precision': [tf.keras.metrics.Precision() for _ in range(num_classes)],
        'recall':    [tf.keras.metrics.Recall() for _ in range(num_classes)],
        'loss':      [tf.keras.metrics.Mean() for _ in range(num_classes)],
        'loss_all': tf.keras.metrics.Mean(),
    }
    val_class_metrics = {
        'accuracy':  [tf.keras.metrics.Accuracy() for _ in range(num_classes)],
        'precision': [tf.keras.metrics.Precision() for _ in range(num_classes)],
        'recall':    [tf.keras.metrics.Recall() for _ in range(num_classes)],
        'loss':      [tf.keras.metrics.Mean() for _ in range(num_classes)],
        'loss_all': tf.keras.metrics.Mean(),
    }

    # 10. Begin the custom training loop.
    data_loading_counter = 0
    for epoch in range(starting_epoch, epochs):
        print("\n############################")
        print(f"Epoch {epoch+1}/{epochs}")
        print("############################")

        current_params = get_parameters_for_epoch(epoch+1, training_schedule)
        try:
            parsed = parse_and_validate_schedule_params(
                current_params=current_params,
                num_classes=num_classes,
                train_indexes=train_indexes
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

        print(f"\nParameters for Epoch {epoch+1}:")
        print(f"  scan_batch_size: {scan_batch_size}")
        print(f"  slice_sub_batch_size: {slice_sub_batch_size}")
        print(f"  accumulate_n_sub_batches: {accumulate_n_sub_batches}")
        print(f"  num_input_slices (2.5D): {num_input_slices}")
        print(f"  num_output_slices (2.5D): {num_output_slices}")
        print(f"  conduct_validation: {conduct_validation}")
        print(f"  validation_frequency: {validation_frequency}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  Tversky alphas: {alpha_vals_list}")
        print(f"  Tversky gamma: {tversky_gamma}")
        print(f"  WCE Loss weight: {wce_weight}")
        print(f"  User specified class_weights: {class_weights}")
        print(f"  class_multiplication_factors: {class_multiplication_factors}")
        print(f"  require_classes: {require_classes}")
        print(f"  epochs_per_new_training_data: {epochs_per_data}\n")

        optimizer.learning_rate.assign(learning_rate)

        data_loading_counter += 1
        if 'X_epoch_data' not in locals() or (data_loading_counter % epochs_per_data == 0):
            data_loading_counter = 0
            print("Loading training data (2.5D) for this epoch...")
            mem = psutil.virtual_memory()
            print(f"System RAM usage before data load: {mem.percent:.2f}%")
            X_epoch_data, y_epoch_data, mask_epoch_data, epoch_sample_names, epoch_class_weights = load_epoch_data(
                scan_indexes=np.random.choice(train_indexes, scan_batch_size, replace=False),
                volume_paths_list=train_volume_paths_list,
                mask_paths=train_mask_paths,
                gt_paths=train_gt_paths,
                slicing_plane=slicing_plane,
                num_input_slices=num_input_slices,
                num_output_slices=num_output_slices,
                num_classes=num_classes,
                class_multiplication_factors=class_multiplication_factors,
                require_classes=require_classes
            )

        if class_weights is not None:
            final_class_weights = []
            for i in range(num_classes):
                cw_value = class_weights[i]
                if math.isnan(cw_value):
                    final_class_weights.append(epoch_class_weights[i])
                else:
                    final_class_weights.append(cw_value)
            print(f"Using mixed user+dynamic class weights: {final_class_weights}")
        else:
            final_class_weights = epoch_class_weights
            print(f"Using dynamic class weights: {final_class_weights}")

        # Build the data generator for training.
        epoch_train_generator = EpochDataGenerator(
            X_epoch_data, y_epoch_data, mask_epoch_data, epoch_sample_names, slice_sub_batch_size
        )
        total_batches = len(epoch_train_generator)
        accumulated_gradients = [tf.zeros_like(tv, dtype=tf.float32) for tv in model.trainable_variables]
        batch_counter = 0

        mem = psutil.virtual_memory()
        print(f"System RAM usage after data load: {mem.percent:.2f}%")

        train_agg = {
            'correct_by_class': np.zeros(num_classes, dtype=np.int64),
            'gt_count_by_class': np.zeros(num_classes, dtype=np.int64),
            'pred_count_by_class': np.zeros(num_classes, dtype=np.int64),
            'total_samples': 0
        }

        print("Training model...")
        for x_batch, y_batch, mask_batch, batch_samples in epoch_train_generator:
            with tf.GradientTape() as tape:
                test_probabilities = model(x_batch, training=True)
                y_batch_one_hot = tf.squeeze(tf.one_hot(y_batch, depth=num_classes), axis=-2)
                loss_value, loss_per_class = combined_focal_tversky_wce_loss(
                    y_true=y_batch_one_hot,
                    y_pred=test_probabilities,
                    mask=mask_batch,
                    class_weights=final_class_weights,
                    alpha_vals=alpha_vals_list,
                    beta_vals=beta_vals_list,
                    gamma=tversky_gamma,
                    wce_weight=wce_weight
                )
            grads = tape.gradient(loss_value, model.trainable_variables)
            for i, grad in enumerate(grads):
                accumulated_gradients[i] += grad

            train_class_metrics['loss_all'].update_state(loss_value)
            for c in range(num_classes):
                train_class_metrics['loss'][c].update_state(loss_per_class[c])

            masked_predictions_filtered, masked_y_batch_filtered = compute_masked_predictions(
                test_probabilities, y_batch, mask_batch, num_classes
            )

            for class_index in range(num_classes):
                y_true_class = (masked_y_batch_filtered == class_index)
                y_pred_class = (masked_predictions_filtered == class_index)
                train_class_metrics['accuracy'][class_index].update_state(y_true_class, y_pred_class)
                train_class_metrics['precision'][class_index].update_state(y_true_class, y_pred_class)
                train_class_metrics['recall'][class_index].update_state(y_true_class, y_pred_class)

            train_agg['total_samples'] += len(masked_y_batch_filtered)
            for c in range(num_classes):
                tp = np.sum((masked_y_batch_filtered == c) & (masked_predictions_filtered == c))
                train_agg['correct_by_class'][c] += tp
                train_agg['gt_count_by_class'][c] += np.sum(masked_y_batch_filtered == c)
                train_agg['pred_count_by_class'][c] += np.sum(masked_predictions_filtered == c)

            batch_counter += 1
            if (batch_counter % accumulate_n_sub_batches == 0) or (batch_counter == total_batches):
                model.optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                accumulated_gradients = [tf.zeros_like(tv) for tv in model.trainable_variables]
            if (batch_counter % print_every_n_subbatches == 0) or (batch_counter == total_batches):
                print(f"Completed {batch_counter}/{total_batches} training batches.")

        # Save a checkpoint at the end of the epoch.
        checkpoint_filename = get_checkpoint_name(epoch + 1)
        model_file_path = os.path.join(model_subdir, checkpoint_filename)
        model.save(model_file_path)
        print(f"Saved model checkpoint to {model_file_path}")
        if IS_NEWER_TF:
            print("Saved in .keras file format.")
        else:
            print("Saved in SavedModel directory format.")

        # --- Log Training Performance (always) ---
        train_class_metrics_for_logging = prepare_class_metrics_for_logging(train_class_metrics)
        train_total_loss = train_class_metrics['loss_all'].result().numpy()
        train_weighted = compute_weighted_macro_metrics(train_agg, num_classes)
        train_wm_prec = train_weighted['weighted_macro_precision']
        train_wm_rec  = train_weighted['weighted_macro_recall']
        train_micro_acc = train_weighted['micro_accuracy']
        train_all_classes_metrics = {
            'accuracy':  train_micro_acc,
            'precision': train_wm_prec,
            'recall':    train_wm_rec,
            'loss':      train_total_loss
        }
        append_metrics_to_file(
            train_metrics_file_path,
            epoch+1,
            train_class_metrics_for_logging,
            all_classes_metrics=train_all_classes_metrics
        )
        print(f"\nEpoch {epoch+1} Train Report:")
        for class_index in range(num_classes):
            accuracy = train_class_metrics['accuracy'][class_index].result().numpy()
            precision = train_class_metrics['precision'][class_index].result().numpy()
            recall = train_class_metrics['recall'][class_index].result().numpy()
            cls_loss = train_class_metrics['loss'][class_index].result().numpy()
            print(f"Class {class_index} - Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, Loss: {cls_loss:.4f}")
        print("ALL_CLASSES (TRAIN) - MicroAcc: {0:.3f}, W-Prec: {1:.3f}, W-Rec: {2:.3f}, Loss: {3:.4f}".format(
            train_micro_acc, train_wm_prec, train_wm_rec, train_total_loss))
        # Reset training metrics.
        for class_index in range(num_classes):
            train_class_metrics['accuracy'][class_index].reset_state()
            train_class_metrics['precision'][class_index].reset_state()
            train_class_metrics['recall'][class_index].reset_state()
            train_class_metrics['loss'][class_index].reset_state()
        train_class_metrics['loss_all'].reset_state()

        # --- Run Validation (if enabled) ---
        if conduct_validation and (epoch % validation_frequency == 0):
            print("Conducting validation (2.5D)...")
            num_val_scans = len(val_mask_paths)
            num_val_batches = int(np.ceil(num_val_scans / scan_batch_size))
            val_agg = {
                'correct_by_class': np.zeros(num_classes, dtype=np.int64),
                'gt_count_by_class': np.zeros(num_classes, dtype=np.int64),
                'pred_count_by_class': np.zeros(num_classes, dtype=np.int64),
                'total_samples': 0
            }
            for batch_num in range(num_val_batches):
                print(f"Validation batch {batch_num+1}/{num_val_batches}...")
                start_idx = batch_num * scan_batch_size
                end_idx = min((batch_num + 1) * scan_batch_size, num_val_scans)
                batch_indexes = np.arange(start_idx, end_idx)
                X_val_data, y_val_data, mask_val_data, mask_z_indices = load_val_data(
                    scan_indexes=batch_indexes,
                    volume_paths_list=val_volume_paths_list,
                    mask_paths=val_mask_paths,
                    gt_paths=val_gt_paths,
                    slicing_plane=slicing_plane,
                    num_input_slices=num_input_slices,
                    num_output_slices=num_output_slices,
                    return_transform_info=False
                )
                val_gen = ValDataGenerator(X_val_data, y_val_data, mask_val_data, slice_sub_batch_size)
                for x_val_batch, y_val_batch_slice, mask_val_batch_slice in val_gen:
                    val_probabilities = model(x_val_batch, training=False)
                    y_val_one_hot = tf.squeeze(tf.one_hot(y_val_batch_slice, depth=num_classes), axis=-2)
                    loss_val, loss_per_class_val = combined_focal_tversky_wce_loss(
                        y_true=y_val_one_hot,
                        y_pred=val_probabilities,
                        mask=mask_val_batch_slice,
                        class_weights=final_class_weights,
                        alpha_vals=alpha_vals_list,
                        beta_vals=beta_vals_list,
                        gamma=tversky_gamma,
                        wce_weight=wce_weight
                    )
                    val_class_metrics['loss_all'].update_state(loss_val)
                    for c in range(num_classes):
                        val_class_metrics['loss'][c].update_state(loss_per_class_val[c])
                    masked_pred_filt, masked_gt_filt = compute_masked_predictions(
                        val_probabilities.numpy(),
                        y_val_batch_slice,
                        mask_val_batch_slice,
                        num_classes
                    )
                    for class_index in range(num_classes):
                        y_true_class = (masked_gt_filt == class_index)
                        y_pred_class = (masked_pred_filt == class_index)
                        val_class_metrics['accuracy'][class_index].update_state(y_true_class, y_pred_class)
                        val_class_metrics['precision'][class_index].update_state(y_true_class, y_pred_class)
                        val_class_metrics['recall'][class_index].update_state(y_true_class, y_pred_class)
                    val_agg['total_samples'] += len(masked_gt_filt)
                    for c in range(num_classes):
                        tp = np.sum((masked_gt_filt == c) & (masked_pred_filt == c))
                        val_agg['correct_by_class'][c] += tp
                        val_agg['gt_count_by_class'][c] += np.sum(masked_gt_filt == c)
                        val_agg['pred_count_by_class'][c] += np.sum(masked_pred_filt == c)
                X_val_data, y_val_data, mask_val_data = None, None, None
                del val_gen
                gc.collect()

            val_class_metrics_for_logging = prepare_class_metrics_for_logging(val_class_metrics)
            val_total_loss = val_class_metrics['loss_all'].result().numpy()
            val_weighted = compute_weighted_macro_metrics(val_agg, num_classes)
            val_wm_prec = val_weighted['weighted_macro_precision']
            val_wm_rec  = val_weighted['weighted_macro_recall']
            val_micro_acc = val_weighted['micro_accuracy']
            val_all_classes_metrics = {
                'accuracy':  val_micro_acc,
                'precision': val_wm_prec,
                'recall':    val_wm_rec,
                'loss':      val_total_loss
            }
            append_metrics_to_file(
                val_metrics_file_path,
                epoch+1,
                val_class_metrics_for_logging,
                all_classes_metrics=val_all_classes_metrics
            )
            print(f"\nEpoch {epoch+1} Validation Report:")
            for class_index in range(num_classes):
                accuracy = val_class_metrics['accuracy'][class_index].result().numpy()
                precision = val_class_metrics['precision'][class_index].result().numpy()
                recall = val_class_metrics['recall'][class_index].result().numpy()
                cls_loss = val_class_metrics['loss'][class_index].result().numpy()
                print(f"Class {class_index} - Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, Loss: {cls_loss:.4f}")
            print("ALL_CLASSES (VAL) - MicroAcc: {0:.3f}, W-Prec: {1:.3f}, W-Rec: {2:.3f}, Loss: {3:.3f}".format(
                val_micro_acc, val_wm_prec, val_wm_rec, val_total_loss))
            for class_index in range(num_classes):
                val_class_metrics['accuracy'][class_index].reset_state()
                val_class_metrics['precision'][class_index].reset_state()
                val_class_metrics['recall'][class_index].reset_state()
                val_class_metrics['loss'][class_index].reset_state()
            val_class_metrics['loss_all'].reset_state()

    print("Training completed.")
    log_file.close()

if __name__ == "__main__":
    train_model()
