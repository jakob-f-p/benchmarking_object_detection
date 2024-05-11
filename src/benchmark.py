# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.utils.misc import get_file_list
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
from mmengine.registry import VISUALIZERS


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("img", help="Image path, include image file, dir and URL.")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--out-dir", default="./output", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--show", action="store_true", help="Show the detection results")
    parser.add_argument("--deploy", action="store_true", help="Switch model to deployment mode")
    parser.add_argument("--tta", action="store_true", help="Whether to use test time augmentation")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Bbox score threshold")
    parser.add_argument("--class-name", nargs="+", type=str, help="Only Save those classes if set")
    parser.add_argument("--to-labelme", action="store_true", help="Output labelme style label file")
    return parser.parse_args(["resources/demo.jpg",
                              "src/demo_config.py",
                              "resources/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth",
                              "--device=cpu"])


def run() -> None:
    args = parse_args()

    if args.to_labelme and args.show:
        msg = "`--to-labelme` or `--show` only can choose one at the same time."
        raise RuntimeError(msg)
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        msg = (
            "config must be a filename or Config object, "
                        f"but got {type(config)}"
        )
        raise TypeError(msg)
    if "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        if "tta_model" not in config:
            "Cannot find ``tta_model`` in config. Can't use tta !"
        if "tta_pipeline" not in config:
            "Cannot find ``tta_pipeline`` in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while "dataset" in test_data_cfg:
            test_data_cfg = test_data_cfg["dataset"]

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if "batch_shapes_cfg" in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get("classes")

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, "bgr", "rgb")

        filename = str(os.path.relpath(file, args.img).replace("/", "_")) if source_type["is_dir"] else Path(file).name
        out_file = None if args.show else Path(args.out_dir) / filename

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        if args.to_labelme:
            # save result to labelme files
            out_file.suffix = ".json"
            continue

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

    if not args.show and not args.to_labelme:
        print_log(f"\nResults have been saved at {Path(args.out_dir).resolve()}")

    elif args.to_labelme:
        print_log(f"\nLabelme format label files had all been saved in {args.out_dir}")


if __name__ == "__main__":
    run()
