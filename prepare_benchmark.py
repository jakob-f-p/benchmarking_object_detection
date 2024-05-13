import json
from argparse import ArgumentParser
from pathlib import Path
from typing import cast, Sequence

from src.download_resources import (DownloadableResource, Resources, download_resources, ModelResource, Resource)
from src.categories import CategoriesFilter, filter_image_annotations_internal

image_annotations_name = "test_image_annotations"
coco2017_dataset = Resources(
    name="MS-COCO dataset (2017)",
    local_root="data/coco",
    resources=[
        DownloadableResource(
            name="test_images",
            remote_path="http://images.cocodataset.org/zips/val2017.zip",
            local_path=Path("val2017/")
        ),
        DownloadableResource(
            name=image_annotations_name,
            remote_path="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            local_path=Path("annotations/instances_val2017.json")
        )
    ]
)

models_default = Resources(
    name="Models",
    local_root="checkpoints",
    resources=[
        ModelResource(
            trained_model=DownloadableResource(
                name="DetectoRS",
                remote_path="https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco"
                            "/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth",
                local_path="detectors_htc_r101_20e_coco.pth",
                full_name="DetectoRS (HTC + ResNet-101)"
            ),
            config=Resource(
                name="DetectoRS-config",
                local_path="detectors/detectors_htc-r101_20e_coco.py"
            )
        ),
        ModelResource(
            trained_model=DownloadableResource(
                name="DINO",
                remote_path="https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl"
                            "/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
                local_path="dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
                full_name="DINO Swin-L 5scale 36e"
            ),
            config=Resource(
                name="DINO-config",
                local_path="dino/dino-5scale_swin-l_8xb2-36e_coco.py",
            )
        ),
        ModelResource(
            trained_model=DownloadableResource(
                name="YOLOX-x",
                remote_path="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco"
                            "/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
                local_path="yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
                full_name="YOLOX-x"
            ),
            config=Resource(
                name="YOLOX-x-config",
                local_path="yolox/yolox_x_8xb8-300e_coco.py",
            )
        )
    ]
)

models_tiny = Resources(
    name="Models-Tiny",
    local_root=models_default.local_root,
    resources=[
        ModelResource(
            trained_model=DownloadableResource(
                name="YOLOX-tiny",
                remote_path="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco"
                            "/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
                local_path="yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
                full_name="YOLOX-tiny"
            ),
            config=Resource(
                name="YOLOX-tiny-config",
                local_path="yolox/yolox_tiny_8xb8-300e_coco.py",
            )
        )
    ]
)


def filter_image_annotations(categories_filter: CategoriesFilter) -> None:
    image_annotations_resource = cast(DownloadableResource,
                                      coco2017_dataset.get_resource_by_name(image_annotations_name))
    image_annotations_path: Path = coco2017_dataset.local_root / image_annotations_resource.local_path
    with open(image_annotations_path) as f:
        image_annotations: dict = json.load(f)

    filter_image_annotations_internal(image_annotations, categories_filter)

    filtered_image_annotations_path = image_annotations_path.with_stem(
        f"{image_annotations_path.stem}_filtered_{categories_filter.name}")
    with open(filtered_image_annotations_path, "w+") as f:
        json.dump(image_annotations, f)
        print(f"Wrote filtered annotations to {filtered_image_annotations_path}")

    image_annotations_resource.add_filtered_resource(categories_filter.name, filtered_image_annotations_path)


def get_full_config_path(root: Path, relative_path: Path) -> Path:
    return (root / ".." / "configs" / relative_path).resolve()


def create_filtered_model_configs(models: Resources, categories_filter: CategoriesFilter) -> None:
    image_annotations_resource = cast(ModelResource, coco2017_dataset.get_resource_by_name(image_annotations_name))
    filtered_annotations: Path = image_annotations_resource.get_filtered_resource(categories_filter.name)

    model: ModelResource
    for model in models:
        relative_config_path: Path = model.config.local_path
        relative_filtered_config_path: Path = relative_config_path.with_stem(
            f"{relative_config_path.stem}_filtered_{categories_filter.name}")
        filtered_config_path: Path = get_full_config_path(models.local_root, relative_filtered_config_path)
        with open(filtered_config_path, "w+") as f:
            f.write(f"_base_ = './{relative_config_path.name}'\n\n"
                    f"val_dataloader = dict(dataset=dict(\n"
                    f"    ann_file='{filtered_annotations.relative_to('data/coco').as_posix()}'\n"
                    f"))\n\n"
                    f"test_dataloader = dict(dataset=dict(\n"
                    f"    ann_file='{filtered_annotations.relative_to('data/coco').as_posix()}'\n"
                    f"))\n"
                    f"val_evaluator = ["
                    f"    dict(\n"
                    f"        type='CocoMetric',\n"
                    f"        ann_file='data/coco/annotations/instances_val2017.json',\n"
                    f"        metric='bbox',\n"
                    f"        metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300',\n"
                    f"                      'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000']\n"
                    f"    )\n"
                    f"]\n"
                    f"test_evaluator = val_evaluator\n"
            )
            print(f"Wrote config with filter to {filtered_config_path}")

        model.add_filtered_resource(categories_filter.name, relative_filtered_config_path)


def main(passed_args: Sequence[str] = None):
    arg_parser = ArgumentParser(description="Download files for benchmarking. May take a while ...")
    arg_parser.add_argument("--tiny",
                            action="store_true",
                            help="download only the YOLOX-tiny dataset (for testing purposes)")
    args = arg_parser.parse_args(passed_args)

    download_resources(coco2017_dataset)

    models: Resources = models_default if not args.tiny else models_tiny
    download_resources(models)

    traffic_categories_filter = CategoriesFilter(
        name="traffic",
        category_names=["umbrella"],
        super_category_names=[]
        # category_names = ["person", "umbrella", "backpack", "suitcase"],
        # super_category_names = ["vehicle", "outdoor"]
    )
    filter_image_annotations(traffic_categories_filter)

    create_filtered_model_configs(models, traffic_categories_filter)


if __name__ == '__main__':
    main()
