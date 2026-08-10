# These all get dynamically assigned by the loaded config files

# Mixed precision (auto prefers bfloat16 when supported)
use_mixed_precision = False
mixed_precision_dtype = "auto"

# DINOv3
use_dinov3_embeddings = False
dinov3_model_name = "dinov3_vitb16"
dinov3_hub_repo = None           # path to local DINOv3 repo clone (for torch.hub)
dinov3_weights = None            # path or URL to DINOv3 weights (for torch.hub)
dinov3_hf_model_id = None        # HuggingFace model ID (alternative to hub)
dinov3_fusion_levels = None      # list of encoder level indices; None = auto (all except 0)
dinov3_hook_blocks = None        # list of ViT block indices; None = auto
dinov3_num_input_channels = None # MRI channels to project to 3; None = use all
dinov3_frozen = True             # freeze DINOv3 backbone during training
dinov3_frozen_epochs = None      # unfreeze DINOv3 after this many epochs (None = stay frozen)

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
architecture_type = "residual_context_unext_25d"
use_se_blocks = False
use_deep_supervision = True
deep_supervision_weights = [0.25, 0.125]
blocks_per_level = 2
context_stem_channels = None
skip_attention_type = "residual"
use_modality_presence_encoding = True
channel_dropout_strategy = "subset"
channel_dropout_subset_probabilities = {
    "full": 0.50,
    "single": 0.25,
    "double": 0.15,
    "required_only": 0.10,
}
use_ema = True
ema_decay = 0.999

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
pretrained_model_load_optimizer = True
pretrained_transfer_mode = "full_checkpoint"
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
