from detectron2.config import CfgNode as CN

__all__ = ["add_mgnet_config"]


def add_mgnet_config(cfg):
    """
    Add config for MGNet.
    """
    # General parameters.
    # Sets torch.backends.cudnn.deterministic variable to increase determinism.
    cfg.CUDNN_DETERMINISTIC = False
    # Store git commit id in config.
    cfg.COMMIT_ID = ""
    # If true, a subdir in cfg.OUTPUT_DIR is created based on current time and config.
    # All output is redirected to the newly created OUTPUT_DIR
    cfg.WRITE_OUTPUT_TO_SUBDIR = True
    # Set either to false to train panoptic or depth only.
    cfg.WITH_PANOPTIC = True
    cfg.WITH_DEPTH = True
    # If set to true, task uncertainty weighting is used as described in the paper.
    cfg.WITH_UNCERTAINTY = True
    # If true, outputs will be visualized during evaluation. This is just for debug purposes.
    cfg.VISUALIZE_EVALUATION = False

    # Solver parameters.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.HEAD_LR_FACTOR = 10.0

    # Specify dataset mappers. String needs to contain the package path as well as the class name.
    cfg.INPUT.TRAIN_DATASET_MAPPER = "mgnet.data.MGNetTrainDatasetMapper"
    cfg.INPUT.TEST_DATASET_MAPPER = "mgnet.data.MGNetTestDatasetMapper"

    # Augmentation parameters.
    # Apply color jitter augmentation with random brightness, contrast, saturation and hue.
    cfg.INPUT.COLOR_JITTER = CN()
    cfg.INPUT.COLOR_JITTER.ENABLED = True
    cfg.INPUT.COLOR_JITTER.BRIGHTNESS = 0.2
    cfg.INPUT.COLOR_JITTER.CONTRAST = 0.2
    cfg.INPUT.COLOR_JITTER.SATURATION = 0.2
    cfg.INPUT.COLOR_JITTER.HUE = 0.05
    # If True, images are randomly padded to the crop size.
    cfg.INPUT.CROP.RANDOM_PAD_TO_CROP_SIZE = True

    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 8
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    cfg.INPUT.IGNORED_CATEGORIES_IN_DEPTH = []

    # Model parameters.
    cfg.MODEL.SIZE_DIVISIBILITY = 32

    cfg.MODEL.GCM = CN()
    cfg.MODEL.GCM.GCM_CHANNELS = 128
    cfg.MODEL.GCM.INIT_METHOD = "xavier"

    # MGNet semantic segmentation head.
    cfg.MODEL.SEM_SEG_HEAD.ARM_CHANNELS = [128, 128]
    cfg.MODEL.SEM_SEG_HEAD.REFINE_CHANNELS = [128, 128]
    cfg.MODEL.SEM_SEG_HEAD.FFM_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    # Layer initialization method, choose from `default`, `xavier`.
    # `default` uses Pytorchs default inits.
    cfg.MODEL.SEM_SEG_HEAD.INIT_METHOD = "xavier"
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`, `ohem`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "ohem"
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    cfg.MODEL.SEM_SEG_HEAD.OHEM_THRESHOLD = 0.7
    cfg.MODEL.SEM_SEG_HEAD.OHEM_N_MIN = 100000

    # MGNet instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "MGNetInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 8
    cfg.MODEL.INS_EMBED_HEAD.ARM_CHANNELS = [128, 128]
    cfg.MODEL.INS_EMBED_HEAD.REFINE_CHANNELS = [128, 128]
    cfg.MODEL.INS_EMBED_HEAD.FFM_CHANNELS = 256
    cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 256
    # Layer initialization method, choose from `default`, `xavier`.
    # `default` uses Pytorchs default inits.
    cfg.MODEL.INS_EMBED_HEAD.INIT_METHOD = "xavier"
    # Loss parameters.
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01

    # MGNet depth head.
    cfg.MODEL.DEPTH_HEAD = CN()
    cfg.MODEL.DEPTH_HEAD.NAME = "MGNetSelfSupervisedDepthHead"
    cfg.MODEL.DEPTH_HEAD.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.DEPTH_HEAD.COMMON_STRIDE = 8
    cfg.MODEL.DEPTH_HEAD.ARM_CHANNELS = [128, 128]
    cfg.MODEL.DEPTH_HEAD.REFINE_CHANNELS = [128, 128]
    cfg.MODEL.DEPTH_HEAD.FFM_CHANNELS = 256
    cfg.MODEL.DEPTH_HEAD.HEAD_CHANNELS = 256
    # Layer initialization method, choose from `default`, `xavier`.
    # `default` uses Pytorchs default inits.
    cfg.MODEL.DEPTH_HEAD.INIT_METHOD = "default"
    # Loss parameters.
    cfg.MODEL.DEPTH_HEAD.MSC_LOSS = True
    cfg.MODEL.DEPTH_HEAD.SSIM_LOSS_WEIGHT = 0.85
    cfg.MODEL.DEPTH_HEAD.PHOTOMETRIC_LOSS_WEIGHT = 1.0
    cfg.MODEL.DEPTH_HEAD.SMOOTHING_LOSS_WEIGHT = 0.001
    cfg.MODEL.DEPTH_HEAD.AUTOMASK_LOSS = True
    # Photometric reduce operation, choose from `min`, `mean`.
    # If AUTOMASK_LOSS is set to true, PHOTOMETRIC_REDUCE_OP is automatically switched to 'min'
    cfg.MODEL.DEPTH_HEAD.PHOTOMETRIC_REDUCE_OP = "min"
    # Padding mode for F.grid_sample, choose from `zeros`, `border`, `reflection`.
    cfg.MODEL.DEPTH_HEAD.PADDING_MODE = "zeros"

    # Post-processing parameters.
    cfg.MODEL.POST_PROCESSING = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.POST_PROCESSING.STUFF_AREA = 2048
    cfg.MODEL.POST_PROCESSING.CENTER_THRESHOLD = 0.3
    cfg.MODEL.POST_PROCESSING.NMS_KERNEL = 7
    # If set to False, MGNet will not use DGC to estimate the depth scale factor
    cfg.MODEL.POST_PROCESSING.USE_DGC_SCALING = True

    # Test parameters
    cfg.TEST.AMP = CN()
    cfg.TEST.AMP.ENABLED = True
    cfg.TEST.MSC_FLIP_EVAL = False
    # Panoptic and depth evaluation are based on cfg.WITH_PANOPTIC and cfg.WITH_DEPTH
    # Semantic and instance evaluation can be enabled separately using these options.
    cfg.TEST.EVAL_SEMANTIC = True
    cfg.TEST.EVAL_INSTANCE = False
    # Minimum and maximum depth in meter. Values outside will be masked out for metric calculations.
    cfg.TEST.MIN_DEPTH = 0.001
    cfg.TEST.MAX_DEPTH = 80.0
