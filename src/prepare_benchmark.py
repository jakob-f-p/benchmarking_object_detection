from argparse import ArgumentParser
from pathlib import Path
from typing import List, Set
from tools.misc.download_dataset import download

import json


class DownloadableResource:
    name: str
    remote_path: str
    local_path: Path
    full_name: str

    def __init__(self, name: str, remote_path: str, local_path, full_name=""):
        self.name = name
        self.remote_path = remote_path
        self.local_path = Path(local_path)
        self.full_name = full_name


class DownloadableResources:
    name: str
    local_destination: Path
    resources: List[DownloadableResource]

    def __init__(self, name, local_destination, resources):
        self.name = name
        self.local_destination = Path(local_destination)
        self.resources = resources

    def __iter__(self):
        return iter(self.resources)

    def full_path(self, resource: DownloadableResource) -> Path:
        return self.local_destination / resource.local_path

    def resource_names(self) -> List[str]:
        return [res.name for res in self]

    def get_resource_by_name(self, name: str) -> DownloadableResource:
        return next(res for res in self if res.name == name)


image_annotations_name = "test_image_annotations"
coco2017_dataset = DownloadableResources(
    name="MS-COCO dataset (2017)",
    local_destination=Path("..") / "data/coco",
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


class CocoCategory:
    id: int
    name: str
    super_category: str

    def __init__(self, id: int, name: str, super_category: str):
        self.id = id
        self.name = name
        self.super_category = super_category

    def __str__(self):
        return f"({self.id}, {self.name}, {self.super_category})"


class CocoCategories:
    categories: List[CocoCategory]
    category_names: List[str]
    super_category_names: List[str]

    def __init__(self, categories: List[CocoCategory]):
        self.categories = categories
        self.category_names = [category.name for category in self.categories]
        self.super_category_names = list(set(category.super_category for category in self.categories))

    @classmethod
    def from_dict(cls, category_list: List[dict]):
        categories: List[CocoCategory] = []
        for category in category_list:
            categories.append(CocoCategory(category["id"],
                                           category["name"],
                                           category["supercategory"]))
        return cls(categories)

    def __iter__(self):
        return iter(self.categories)

    def __str__(self):
        return str([category.__str__() for category in self.categories])

    def apply_filter(self, categories_filter):
        filtered_categories: List[CocoCategory] = [category for category in self.categories
                                                   if category.id in categories_filter.category_ids(self)]
        return CocoCategories(filtered_categories)

    def category_ids(self) -> List[int]:
        return [category.id for category in self.categories]


class CategoriesFilter:
    name: str
    category_names: List[str]
    super_category_names: List[str]

    def __init__(self, name: str, category_names: List[str], super_category_names: List[str]):
        self.name = name
        self.category_names = category_names
        self.super_category_names = super_category_names

    def __str__(self):
        return (f"category names: {self.category_names} \n"
                f"super category names: {self.super_category_names} \n"
                f"categories: {str([category.__str__() for category in self.allowed_categories])}")

    def category_ids(self, categories: CocoCategories) -> List[int]:
        return [category.id for category in categories
                if category.name in self.category_names
                or category.super_category in self.super_category_names]


models = DownloadableResources(
    name="Models",
    local_destination=Path("..") / "checkpoints",
    resources=[
        DownloadableResource(
            name="DetectoRS",
            remote_path="https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco"
                        "/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth",
            local_path=Path("detectors_htc_r101_20e_coco.pth"),
            full_name="DetectoRS (HTC + ResNet-101)"
        ),
        DownloadableResource(
            name="DINO",
            remote_path="https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl"
                        "/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
            local_path=Path("dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"),
            full_name="DINO Swin-L 5scale 36e"
        ),
        DownloadableResource(
            name="YOLOX-x",
            remote_path="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco"
                        "/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
            local_path=Path("yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"),
            full_name="YOLOX-x"
        )
    ]
)

models_tiny = DownloadableResources(
    name="Models-Tiny",
    local_destination=models.local_destination,
    resources=[
        DownloadableResource(
            name="YOLOX-tiny",
            remote_path="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco"
                        "/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
            local_path=Path("yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"),
            full_name="YOLOX-tiny"
        )
    ]
)


def get_non_downloaded_resources(resources: DownloadableResources, root_path: Path) -> List[DownloadableResource]:
    return [res for res in resources if not (root_path / res.local_path).exists()]


def download_resources(resources: DownloadableResources) -> None:
    print(f"Checking {resources.name} resources...")

    root_path = resources.local_destination
    if not root_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)

    non_downloaded_resources = get_non_downloaded_resources(resources, root_path)
    if non_downloaded_resources:
        non_downloaded_names = [res.name for res in non_downloaded_resources]
        non_downloaded_files = [res.remote_path for res in non_downloaded_resources]

        print(f"Downloading resources: {non_downloaded_names}")

        download(non_downloaded_files, dir=root_path, unzip=True, delete=True)
    else:
        names = [res.name for res in resources]
        print(f"Resources already downloaded: {names}")


def filter_image_annotations(image_annotations: dict, categories_filter: CategoriesFilter) -> None:
    previous_number_of_images: int = len(image_annotations["images"])

    categories = CocoCategories.from_dict(image_annotations["categories"])
    filtered_image_categories = categories.apply_filter(categories_filter)

    image_annotations["categories"] = [category for category in image_annotations["categories"]
                                       if category["id"] in filtered_image_categories.category_ids()]
    image_annotations["annotations"] = [annotation for annotation in image_annotations["annotations"]
                                        if annotation["category_id"] in filtered_image_categories.category_ids()]
    remaining_image_ids: Set[int] = set([annotation["image_id"] for annotation in image_annotations["annotations"]])
    image_annotations["images"] = [image for image in image_annotations["images"]
                                   if image["id"] in remaining_image_ids]

    print(f"Filter '{categories_filter.name}' applied. Reduced number of images from {previous_number_of_images}"
          f" to {len(image_annotations['images'])}")


def main():
    arg_parser = ArgumentParser(description="Download files for benchmarking. May take a while ...")
    arg_parser.add_argument("--tiny",
                            action="store_true",
                            help="download only the YOLOX-tiny dataset (for testing purposes)")
    args = arg_parser.parse_args()

    download_resources(coco2017_dataset)

    download_resources(models if not args.tiny else models_tiny)

    image_annotations_resource: DownloadableResource = coco2017_dataset.get_resource_by_name(image_annotations_name)
    image_annotations_path: Path = coco2017_dataset.full_path(image_annotations_resource)
    with open(image_annotations_path) as f:
        image_annotations: dict = json.load(f)

    traffic_categories_filter = CategoriesFilter(
        name="traffic",
        category_names=["person", "umbrella", "backpack", "suitcase"],
        super_category_names=["vehicle", "outdoor"]
    )
    filter_image_annotations(image_annotations, traffic_categories_filter)

    filtered_image_annotations_path = image_annotations_path.with_stem(
        f"{image_annotations_path.stem}_filtered_{traffic_categories_filter.name}")
    with open(filtered_image_annotations_path, "w+") as f:
        json.dump(image_annotations, f)
        print(f"Wrote filtered annotations to {filtered_image_annotations_path}")


if __name__ == '__main__':
    main()
