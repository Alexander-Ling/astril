#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: visualize_model
Description: Loads a pre-trained model and logs its TensorFlow graph for visualization with TensorBoard.
Users only need to provide the model path and, if needed, the training config file to auto-extract model parameters.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import configparser

# Import custom objects needed for model loading.
from astril.model_architecture import DynamicAttentionResUNet, ResidualConvBlock, AttentionBlock
# Import the helper that parses training config parameters.
from astril.create_segmentation_config import parse_train_config_for_model_parameters

# Define custom objects for Keras model loading.
custom_objects_dict = {
    'ResidualConvBlock': ResidualConvBlock,
    'AttentionBlock': AttentionBlock,
    'DynamicAttentionResUNet': DynamicAttentionResUNet
}

def build_model_from_train_config(train_config_path):
    """
    Rebuilds the model architecture from the training config file.
    Uses parameters extracted from the config—such as num_input_slices and minimum_height_width—to compute input channels.
    Returns both the built model and the parsed parameters.
    """
    cp = configparser.ConfigParser()
    cp.read(train_config_path)
    cfg = cp["DEFAULT"]
    
    # Extract key training parameters using the helper.
    params = parse_train_config_for_model_parameters(train_config_path)
    
    # Determine number of channels; if not provided, try to infer from "image_paths_files".
    num_channels_val = cfg.get("num_channels", None)
    if num_channels_val is None or num_channels_val.strip() == "":
        ips = cfg.get("image_paths_files", "")
        if ips.strip():
            num_channels = len(ips.split(","))
        else:
            num_channels = 1  # fallback default
    else:
        num_channels = int(num_channels_val)
        
    # Calculate the number of input channels.
    input_channels = num_channels * params["num_input_slices"]
    
    # Get additional model parameters.
    base_num_filters = cfg.getint("base_num_filters", fallback=32)
    encoder_level_factors = [int(x.strip()) for x in cfg.get("encoder_level_factors", fallback="1,2,4,8").split(",") if x.strip()]
    center_depth = cfg.getint("center_depth", fallback=1)
    
    # Build and compile the model.
    model = DynamicAttentionResUNet(
        input_channels=input_channels,
        base_num_filters=base_num_filters,
        encoder_level_factors=encoder_level_factors,
        num_output_slices=params["num_output_slices"],
        out_channels=params["num_classes"],
        center_depth=center_depth
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Save the input channels as an attribute for later use.
    model.input_channels = input_channels
    
    return model, params

def load_model_for_visualization(model_path, train_config_path=None):
    """
    Loads a pre-trained model. If the model file contains only weights and a training config is provided,
    rebuilds the model architecture and loads the weights.
    Returns the loaded model and the parsed parameters (if available).
    """
    try:
        model = load_model(model_path, custom_objects=custom_objects_dict, compile=False)
        params = None
        if train_config_path:
            # Parse parameters from the training config to help define the dummy input shape.
            _, params = build_model_from_train_config(train_config_path)
        return model, params
    except ValueError as e:
        if "No model config found" in str(e) and train_config_path is not None:
            # Rebuild the model from the training config and load weights.
            model, params = build_model_from_train_config(train_config_path)
            dummy_hw = params["minimum_height_width"]
            dummy_input = tf.zeros((1, dummy_hw, dummy_hw, model.input_channels))
            _ = model(dummy_input)  # Run a forward pass to initialize model variables.
            model.load_weights(model_path)
            return model, params
        else:
            raise

def visualize_model(model, log_dir, dummy_input_shape):
    """
    Logs the TensorFlow computation graph of the model by running a dummy forward pass.
    The forward pass is wrapped in a @tf.function so that a concrete function is generated and the graph is captured.
    """
    # Create a TensorBoard log writer.
    writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=True)
    
    # Create dummy input.
    dummy_input = tf.zeros(dummy_input_shape)
    
    # Wrap the model call in a tf.function to force graph tracing.
    @tf.function
    def run_model(x):
        return model(x)
    
    # Call the function to generate a concrete function.
    _ = run_model(dummy_input)
    
    # Export the traced graph.
    with writer.as_default():
        tf.summary.trace_export(
            name="model_trace",
            step=0,
            profiler_outdir=log_dir
        )
    print(f"Model graph has been saved to '{log_dir}'.")
    print(f"Run 'tensorboard --logdir {log_dir}' to visualize the graph.")

def main():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained model and visualize its TensorFlow graph with TensorBoard.\n"
                    "Provide the model path and, if needed, the training config file to auto-extract model parameters."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the pre-trained model file (directory, .keras, or .h5).")
    parser.add_argument("--train_config", default=None,
                        help="Path to the training configuration file (if the model file contains only weights).")
    parser.add_argument("--log_dir", default="./logs",
                        help="Directory to store TensorBoard logs.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the dummy input tensor.")
    args = parser.parse_args()
    
    # Load the pre-trained model.
    model, params = load_model_for_visualization(args.model_path, args.train_config)
    
    # Determine dummy input dimensions:
    # Prefer to get the dimensions from the model's serving signature if available.
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        serving_fn = model.signatures["serving_default"]
        # Extract height and channels from the serving signature.
        input_spec = serving_fn.inputs[0]
        dummy_hw = input_spec.shape[1]
        in_channels = input_spec.shape[-1]
    elif params is not None:
        dummy_hw = params["minimum_height_width"]
        # Compute channels from the training config.
        cp = configparser.ConfigParser()
        cp.read(args.train_config)
        cfg = cp["DEFAULT"]
        num_channels_val = cfg.get("num_channels", None)
        if num_channels_val is None or num_channels_val.strip() == "":
            ips = cfg.get("image_paths_files", "")
            if ips.strip():
                num_channels = len(ips.split(","))
            else:
                num_channels = 1
        else:
            num_channels = int(num_channels_val)
        in_channels = num_channels * params["num_input_slices"]
    else:
        # Fallback defaults.
        dummy_hw = 128
        in_channels = 1

    dummy_input_shape = (args.batch_size, dummy_hw, dummy_hw, in_channels)
    
    print(f"Using dummy input shape: {dummy_input_shape}")
    
    # Visualize the model graph using TensorBoard.
    visualize_model(model, args.log_dir, dummy_input_shape)

if __name__ == "__main__":
    main()
