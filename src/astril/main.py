import os
import argparse
import configparser
import astril.config as config
from pathlib import Path

def parse_train_parameters(config_file_path):
    """
    Parses the train_parameters.cfg file and sets the parameters in the config module.
    """
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(config_file_path)

    config.output_dir = cfg_parser.get("DEFAULT", "output_dir")
    config.n_cores = cfg_parser.getint("DEFAULT", "n_cores")
    config.slicing_plane = cfg_parser.get("DEFAULT", "slicing_plane")
    config.image_paths_files = cfg_parser.get("DEFAULT", "image_paths_files").split(',')
    config.gt_paths_file = cfg_parser.get("DEFAULT", "ground_truth_paths_files")
    config.mask_paths_file = cfg_parser.get("DEFAULT", "mask_paths_files")
    config.num_classes = cfg_parser.getint("DEFAULT", "num_classes")
    config.epochs = cfg_parser.getint("DEFAULT", "epochs")
    config.num_input_slices = cfg_parser.getint("DEFAULT", "num_input_slices")
    config.num_output_slices = cfg_parser.getint("DEFAULT", "num_output_slices")
    config.training_schedule_file = cfg_parser.get("DEFAULT", "training_schedule_file")
    config.pretrained_model_path = cfg_parser.get("DEFAULT", "pretrained_model_path")
    config.print_every_n_subbatches = cfg_parser.getint("DEFAULT", "print_every_n_subbatches")
    config.minimum_height_width = cfg_parser.getint("DEFAULT", "minimum_height_width")
    config.num_channels = len(config.image_paths_files)
    if cfg_parser.has_option("DEFAULT", "base_num_filters"):
        config.base_num_filters = cfg_parser.getint("DEFAULT", "base_num_filters")
    else:
        config.base_num_filters = 32
    if cfg_parser.has_option("DEFAULT", "center_depth"):
        config.center_depth = cfg_parser.getint("DEFAULT", "center_depth")
    else:
        config.center_depth = 1
    if cfg_parser.has_option("DEFAULT", "encoder_level_factors"):
        factors_str = cfg_parser.get("DEFAULT", "encoder_level_factors")
        config.encoder_level_factors = [int(x.strip()) for x in factors_str.split(",") if x.strip()]
    else:
        config.encoder_level_factors = [1, 2, 4, 8]

    if cfg_parser.has_option("DEFAULT", "val_image_paths_files"):
        config.val_image_paths_files = cfg_parser.get("DEFAULT", "val_image_paths_files").split(',')
    if cfg_parser.has_option("DEFAULT", "val_ground_truth_paths_files"):
        config.val_gt_paths_file = cfg_parser.get("DEFAULT", "val_ground_truth_paths_files")
    if cfg_parser.has_option("DEFAULT", "val_mask_paths_files"):
        config.val_mask_paths_file = cfg_parser.get("DEFAULT", "val_mask_paths_files")

def main():
    parser = argparse.ArgumentParser(description="MRI slice-based segmentation training script.")
    parser.add_argument("--config", required=True, help="Path to train_parameters.cfg file.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--n_cores", type=int, help="Override the number of CPU cores.")
    parser.add_argument("--output_dir", type=str, help="Override the output directory.")
    parser.add_argument("--slicing_plane", type=str, choices=["axial", "sagittal", "coronal"], help="Override the slicing plane.")
    parser.add_argument("--training_schedule_file", type=str, help="Override the training schedule file path.")
    parser.add_argument("--print_every_n_subbatches", type=int, help="Override the subbatch logging frequency.")
    parser.add_argument("--minimum_height_width", type=int, help="Override the minimum height and width required for training slices (in pixels).")

    args = parser.parse_args()
    parse_train_parameters(args.config)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.n_cores is not None:
        config.n_cores = args.n_cores
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.slicing_plane is not None:
        config.slicing_plane = args.slicing_plane
    if args.training_schedule_file is not None:
        config.training_schedule_file = args.training_schedule_file
    if args.print_every_n_subbatches is not None:
        config.print_every_n_subbatches = args.print_every_n_subbatches
    if args.minimum_height_width is not None:
        config.minimum_height_width = args.minimum_height_width

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from .train import train_model
    train_model()

if __name__ == "__main__":
    main()
