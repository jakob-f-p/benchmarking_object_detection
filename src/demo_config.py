
_backend_args = None
_multiscale_resize_transforms = [
    {"transforms": [
        {"scale": (640, 640), "type": "YOLOv5KeepRatioResize"},
        {
            "allow_scale_up": False,
            "pad_val": {"img": 114},
            "scale": (640, 640),
            "type": "LetterResize"},
    ],
        "type": "Compose" },
    {
        "transforms": [
            {"scale": (320, 320), "type": "YOLOv5KeepRatioResize"},
            {
                "allow_scale_up": False,
                "pad_val": {"img": 114},
                "scale": (320, 320),
                "type": "LetterResize"},
        ],
        "type": "Compose"},
    {
        "transforms": [
            {"scale": (960, 960), "type": "YOLOv5KeepRatioResize"},
            {
                "allow_scale_up": False,
                "pad_val": {"img": 114},
                "scale": (960, 960),
                "type": "LetterResize"},
        ],
        "type": "Compose"},
]
affine_scale = 0.5
albu_train_transforms = [
    {"p": 0.01, "type": "Blur"},
    {"p": 0.01, "type": "MedianBlur"},
    {"p": 0.01, "type": "ToGray"},
    {"p": 0.01, "type": "CLAHE"},
]
anchors = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]
backend_args = None
base_lr = 0.01
batch_shapes_cfg = {
    "batch_size": 1,
    "extra_pad_ratio": 0.5,
    "img_size": 640,
    "size_divisor": 32,
    "type": "BatchShapePolicy"}
custom_hooks = [
    {
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0001,
        "priority": 49,
        "strict_load": False,
        "type": "EMAHook",
        "update_buffers": True},
]
data_root = "data/coco/"
dataset_type = "YOLOv5CocoDataset"
deepen_factor = 0.33
default_hooks = {
    "checkpoint": {
        "interval": 10, "max_keep_ckpts": 3, "save_best": "auto",
        "type": "CheckpointHook"},
    "logger": {"interval": 50, "type": "LoggerHook"},
    "param_scheduler": {
        "lr_factor": 0.01,
        "max_epochs": 300,
        "scheduler_type": "linear",
        "type": "YOLOv5ParamSchedulerHook"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "timer": {"type": "IterTimerHook"},
    "visualization": {"type": "mmdet.DetVisualizationHook"}}
default_scope = "mmyolo"
env_cfg = {
    "cudnn_benchmark": True,
    "dist_cfg": {"backend": "nccl"},
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0}}
img_scale = (640, 640)
img_scales = [(640, 640), (320, 320), (960, 960)]
load_from = None
log_level = "INFO"
log_processor = {"by_epoch": True, "type": "LogProcessor", "window_size": 50}
loss_bbox_weight = 0.05
loss_cls_weight = 0.5
loss_obj_weight = 1.0
lr_factor = 0.01
max_epochs = 300
max_keep_ckpts = 3
model = {
    "backbone": {
        "act_cfg": {"inplace": True, "type": "SiLU"},
        "deepen_factor": 0.33,
        "norm_cfg": {"eps": 0.001, "momentum": 0.03, "type": "BN"},
        "type": "YOLOv5CSPDarknet",
        "widen_factor": 0.5},
    "bbox_head": {
        "head_module": {
            "featmap_strides": [8, 16, 32],
            "in_channels": [256, 512, 1024],
            "num_base_priors": 3,
            "num_classes": 80,
            "type": "YOLOv5HeadModule",
            "widen_factor": 0.5},
        "loss_bbox": {
            "bbox_format": "xywh",
            "eps": 1e-07,
            "iou_mode": "ciou",
            "loss_weight": 0.05,
            "reduction": "mean",
            "return_iou": True,
            "type": "IoULoss"},
        "loss_cls": {
            "loss_weight": 0.5,
            "reduction": "mean",
            "type": "mmdet.CrossEntropyLoss",
            "use_sigmoid": True},
        "loss_obj": {
            "loss_weight": 1.0,
            "reduction": "mean",
            "type": "mmdet.CrossEntropyLoss",
            "use_sigmoid": True},
        "obj_level_weights": [4.0, 1.0, 0.4 ],
        "prior_generator": {
            "base_sizes": [
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)],
            ],
            "strides": [8, 16, 32],
            "type": "mmdet.YOLOAnchorGenerator"},
        "prior_match_thr": 4.0,
        "type": "YOLOv5Head"},
    "data_preprocessor": {
        "bgr_to_rgb": True,
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "type": "YOLOv5DetDataPreprocessor"},
    "neck": {
        "act_cfg": {"inplace": True, "type": "SiLU"},
        "deepen_factor": 0.33,
        "in_channels": [256, 512, 1024],
        "norm_cfg": {"eps": 0.001, "momentum": 0.03, "type": "BN"},
        "num_csp_blocks": 3,
        "out_channels": [256, 512, 1024],
        "type": "YOLOv5PAFPN",
        "widen_factor": 0.5},
    "test_cfg": {
        "max_per_img": 300,
        "multi_label": True,
        "nms": {"iou_threshold": 0.65, "type": "nms"},
        "nms_pre": 30000,
        "score_thr": 0.001},
    "type": "YOLODetector"}
model_test_cfg = {
    "max_per_img": 300,
    "multi_label": True,
    "nms": {"iou_threshold": 0.65, "type": "nms"},
    "nms_pre": 30000,
    "score_thr": 0.001}
norm_cfg = {"eps": 0.001, "momentum": 0.03, "type": "BN"}
num_classes = 80
num_det_layers = 3
obj_level_weights = [
    4.0,
    1.0,
    0.4,
]
optim_wrapper = {
    "constructor": "YOLOv5OptimizerConstructor",
    "optimizer": {
        "batch_size_per_gpu": 16,
        "lr": 0.01,
        "momentum": 0.937,
        "nesterov": True,
        "type": "SGD",
        "weight_decay": 0.0005},
    "type": "OptimWrapper"}
param_scheduler = None
persistent_workers = True
pre_transform = [
    {"backend_args": None, "type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
]
prior_match_thr = 4.0
resume = False
save_checkpoint_intervals = 10
strides = [8, 16, 32 ]
test_cfg = {"type": "TestLoop"}
test_dataloader = {
    "batch_size": 1,
    "dataset": {
        "ann_file": "annotations/instances_val2017.json",
        "batch_shapes_cfg": {
            "batch_size": 1,
            "extra_pad_ratio": 0.5,
            "img_size": 640,
            "size_divisor": 32,
            "type": "BatchShapePolicy"},
        "data_prefix": {"img": "val2017/"},
        "data_root": "data/coco/",
        "pipeline": [
            {"backend_args": None, "type": "LoadImageFromFile"},
            {"scale": (640, 640), "type": "YOLOv5KeepRatioResize"},
            {
                "allow_scale_up": False,
                "pad_val": {"img": 114},
                "scale": (640, 640),
                "type": "LetterResize"},
            {"_scope_": "mmdet", "type": "LoadAnnotations", "with_bbox": True},
            {
                "meta_keys": (
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "pad_param",
                ),
                "type": "mmdet.PackDetInputs"},
        ],
        "test_mode": True,
        "type": "YOLOv5CocoDataset"},
    "drop_last": False,
    "num_workers": 2,
    "persistent_workers": True,
    "pin_memory": True,
    "sampler": {"shuffle": False, "type": "DefaultSampler"}}
test_evaluator = {
    "ann_file": "data/coco/annotations/instances_val2017.json",
    "metric": "bbox",
    "proposal_nums": (100, 1, 10),
    "type": "mmdet.CocoMetric"}
test_pipeline = [
    {"backend_args": None, "type": "LoadImageFromFile"},
    {"scale": (640, 640), "type": "YOLOv5KeepRatioResize"}, {
        "allow_scale_up": False,
        "pad_val": {"img": 114},
        "scale": (640, 640),
        "type": "LetterResize"},
    {"_scope_": "mmdet", "type": "LoadAnnotations", "with_bbox": True},
    {
        "meta_keys": (
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
        ),
        "type": "mmdet.PackDetInputs"},
]
train_ann_file = "annotations/instances_train2017.json"
train_batch_size_per_gpu = 16
train_cfg = {"max_epochs": 300, "type": "EpochBasedTrainLoop", "val_interval": 10}
train_data_prefix = "train2017/"
train_dataloader = {
    "batch_size": 16,
    "collate_fn": {"type": "yolov5_collate"},
    "dataset": {
        "ann_file": "annotations/instances_train2017.json",
        "data_prefix": {"img": "train2017/"},
        "data_root": "data/coco/",
        "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
        "pipeline": [
            {"backend_args": None, "type": "LoadImageFromFile"},
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "img_scale": (640, 640),
                "pad_val": 114.0,
                "pre_transform": [
                    {"backend_args": None, "type": "LoadImageFromFile"},
                    {"type": "LoadAnnotations", "with_bbox": True},
                ],
                "type": "Mosaic"},
            {
                "border": (-320, -320),
                "border_val": (114, 114, 114),
                "max_rotate_degree": 0.0,
                "max_shear_degree": 0.0,
                "scaling_ratio_range": (0.5, 1.5),
                "type": "YOLOv5RandomAffine"},
            {
                "bbox_params": {
                    "format": "pascal_voc",
                    "label_fields": [
                        "gt_bboxes_labels",
                        "gt_ignore_flags",
                    ],
                    "type": "BboxParams"},
                "keymap": {"gt_bboxes": "bboxes", "img": "image"},
                "transforms": [
                    {"p": 0.01, "type": "Blur"},
                    {"p": 0.01, "type": "MedianBlur"},
                    {"p": 0.01, "type": "ToGray"},
                    {"p": 0.01, "type": "CLAHE"},
                ],
                "type": "mmdet.Albu"},
            {"type": "YOLOv5HSVRandomAug"},
            {"prob": 0.5, "type": "mmdet.RandomFlip"},
            {
                "meta_keys": (
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "flip",
                    "flip_direction",
                ),
                "type": "mmdet.PackDetInputs"},
        ],
        "type": "YOLOv5CocoDataset"},
    "num_workers": 8,
    "persistent_workers": True,
    "pin_memory": True,
    "sampler": {"shuffle": True, "type": "DefaultSampler"}}
train_num_workers = 8
train_pipeline = [
    {"backend_args": None, "type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {
        "img_scale": (640, 640),
        "pad_val": 114.0,
        "pre_transform": [
            {"backend_args": None, "type": "LoadImageFromFile"},
            {"type": "LoadAnnotations", "with_bbox": True},
        ],
        "type": "Mosaic"},
    {
        "border": (-320, -320),
        "border_val": (114, 114, 114),
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (0.5, 1.5),
        "type": "YOLOv5RandomAffine"},
    {
        "bbox_params": {
            "format": "pascal_voc",
            "label_fields": [
                "gt_bboxes_labels",
                "gt_ignore_flags",
            ],
            "type": "BboxParams"},
        "keymap": {"gt_bboxes": "bboxes", "img": "image"},
        "transforms": [
            {"p": 0.01, "type": "Blur"},
            {"p": 0.01, "type": "MedianBlur"},
            {"p": 0.01, "type": "ToGray"},
            {"p": 0.01, "type": "CLAHE"},
        ],
        "type": "mmdet.Albu"},
    {"type": "YOLOv5HSVRandomAug"},
    {"prob": 0.5, "type": "mmdet.RandomFlip"},
    {
        "meta_keys": (
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
        ),
        "type": "mmdet.PackDetInputs"},
]
tta_model = {
    "tta_cfg": {"max_per_img": 300, "nms": {"iou_threshold": 0.65, "type": "nms"}},
    "type": "mmdet.DetTTAModel"}
tta_pipeline = [
    {"backend_args": None, "type": "LoadImageFromFile"},
    {
        "transforms": [
            [
                {
                    "transforms": [
                        {"scale": (640, 640), "type": "YOLOv5KeepRatioResize"},
                        {
                            "allow_scale_up": False,
                            "pad_val": {"img": 114},
                            "scale": (640, 640),
                            "type": "LetterResize"},
                    ],
                    "type": "Compose"},
                {
                    "transforms": [
                        {"scale": (320, 320), "type": "YOLOv5KeepRatioResize"},
                        {
                            "allow_scale_up": False,
                            "pad_val": {"img": 114},
                            "scale": (320, 320),
                            "type": "LetterResize"},
                    ],
                    "type": "Compose"},
                {
                    "transforms": [
                        {"scale": (960, 960), "type": "YOLOv5KeepRatioResize"},
                        {
                            "allow_scale_up": False,
                            "pad_val": {"img": 114},
                            "scale": (960, 960),
                            "type": "LetterResize"},
                    ],
                    "type": "Compose"},
            ],
            [
                {"prob": 1.0, "type": "mmdet.RandomFlip"},
                {"prob": 0.0, "type": "mmdet.RandomFlip"},
            ],
            [
                {"type": "mmdet.LoadAnnotations", "with_bbox": True},
            ],
            [
                {
                    "meta_keys": (
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                        "flip",
                        "flip_direction",
                    ),
                    "type": "mmdet.PackDetInputs"},
            ],
        ],
        "type": "TestTimeAug"},
]
val_ann_file = "annotations/instances_val2017.json"
val_batch_size_per_gpu = 1
val_cfg = {"type": "ValLoop"}
val_data_prefix = "val2017/"
val_dataloader = {
    "batch_size": 1,
    "dataset": {
        "ann_file": "annotations/instances_val2017.json",
        "batch_shapes_cfg": {
            "batch_size": 1,
            "extra_pad_ratio": 0.5,
            "img_size": 640,
            "size_divisor": 32,
            "type": "BatchShapePolicy"},
        "data_prefix": {"img": "val2017/"},
        "data_root": "data/coco/",
        "pipeline": [
            {"backend_args": None, "type": "LoadImageFromFile"},
            {"scale": (640, 640), "type": "YOLOv5KeepRatioResize"},
            {
                "allow_scale_up": False,
                "pad_val": {"img": 114},
                "scale": (640, 640),
                "type": "LetterResize"},
            {"_scope_": "mmdet", "type": "LoadAnnotations", "with_bbox": True},
            {
                "meta_keys": (
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "pad_param",
                ),
                "type": "mmdet.PackDetInputs"},
        ],
        "test_mode": True,
        "type": "YOLOv5CocoDataset"},
    "drop_last": False,
    "num_workers": 2,
    "persistent_workers": True,
    "pin_memory": True,
    "sampler": {"shuffle": False, "type": "DefaultSampler"}}
val_evaluator = {
    "ann_file": "data/coco/annotations/instances_val2017.json",
    "metric": "bbox",
    "proposal_nums": (
        100,
        1,
        10,
    ),
    "type": "mmdet.CocoMetric"}
val_num_workers = 2
vis_backends = [
    {"type": "LocalVisBackend"},
]
visualizer = {
    "name": "visualizer",
    "type": "mmdet.DetLocalVisualizer",
    "vis_backends": [
        {"type": "LocalVisBackend"},
    ]}
weight_decay = 0.0005
widen_factor = 0.5
