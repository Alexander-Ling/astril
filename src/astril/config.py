# These all get dynamically assigned by the loaded config files

# Mixed precision (float16 for RTX tensor cores)
use_mixed_precision = False

# BrainIAC
use_brainiac_embeddings = False
brainiac_weights_path = None        # path to .ckpt if user supplies manually
brainiac_embedding_type = "encoder_fusion"
brainiac_feature_paths_files = None
val_brainiac_feature_paths_files = None
brainiac_encoder_input_channels = 0
brainiac_encode_channels = "all"
brainiac_encode_channel_indices = None

# Architecture flags
use_se_blocks = False
use_deep_supervision = False
deep_supervision_weights = None     # list of floats e.g. [0.5, 0.25]

# Augmentation
use_flip_augmentation = False
use_intensity_augmentation = False
intensity_augmentation_strength = 0.1
use_rotation_augmentation = False
rotation_degrees = 10.0

output_dir = None
n_cores = None
dataloader_num_workers = None
dataloader_prefetch_factor = 4
dataloader_persistent_workers = True
slicing_plane = None
image_paths_files = None         # training channels file list (comma?separated list from create_config_files)
gt_paths_file = None             # training ground truth file
mask_paths_file = None           # training mask file
val_image_paths_files = None     # validation channels file list (list of paths)
val_gt_paths_file = None         # validation ground truth file
val_mask_paths_file = None       # validation mask file
num_classes = None
epochs = None
num_input_slices = None
num_output_slices = None
training_schedule_file = None
pretrained_model_path = None
print_every_n_subbatches = None
minimum_height_width = None
num_channels = None
base_num_filters = None
encoder_level_factors = None
center_depth = None
channel_names = None
optional_channels = []
channel_dropout_probabilities = {}
allow_missing_optional_channels = False
missing_channel_fill = "zero"
