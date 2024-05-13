import os
from argparse import ArgumentParser
from typing import List

import prepare_benchmark
from src.download_resources import ModelResource, Resources
from prepare_benchmark import models_default, models_tiny, get_full_config_path


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to test.",
                        choices=["DetectoRS", "DINO", "YOLOX-x", "YOLOX-tiny"])
    parser.add_argument("--filter", type=str, help="Name of filter to apply", choices=["traffic"])
    parser.add_argument("--show-images", action="store_true",
                        help="show output images live", dest="show_images")
    parser.add_argument("--save-images", action="store_true",
                        help="save output images", dest="save_images")
    args = parser.parse_args()

    model: ModelResource = next((model for model in models_default if model.name == args.model_name), None)
    models: Resources = models_default
    if model is None:
        model = next((model for model in models_tiny if model.name == args.model_name), None)
        models = models_tiny

    if model is None:
        raise ValueError("Model not found.")

    prepare_args: List[str] = []
    if models == models_tiny:
        prepare_args.append("--tiny")

    prepare_benchmark.main(prepare_args)

    config_path = get_full_config_path(models.local_root,
                                       model.config.local_path
                                       if args.filter is None else model.filtered_resources[args.filter])
    model_path = models.local_root / model.trained_model.local_path

    if args.show_images and args.save_images:
        raise ValueError("Choose at most one of --show-images and --save-images")

    call_test_script = (f"python src/tools/test.py {config_path.as_posix()} {model_path.as_posix()}"
                        f" {'--show' if args.show_images else ('--show-dir img' if args.save_images else '')}")
    print(call_test_script)
    os.system(call_test_script)


if __name__ == "__main__":
    run()
