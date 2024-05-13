from pathlib import Path
from src.tools.misc.download_dataset import download
from typing import List, Dict, cast


class Resource:
    name: str
    local_path: Path
    full_name: str
    filtered_resources: Dict[str, Path]

    def __init__(self, name: str, local_path, filtered_resources: Dict[str, Path] = None, full_name: str = ""):
        self.name = name
        self.local_path = Path(local_path)
        if filtered_resources is None:
            filtered_resources = {}
        self.filtered_resources = filtered_resources
        self.full_name = full_name

    def add_filtered_resource(self, name: str, local_path: Path):
        self.filtered_resources[name] = local_path

    def get_filtered_resource(self, name: str) -> Path:
        return self.filtered_resources[name]


class DownloadableResource(Resource):
    remote_path: str

    def __init__(self, name: str, remote_path: str, local_path, filtered_resources=None, full_name=""):
        super().__init__(name, local_path, filtered_resources, full_name)
        self.remote_path = remote_path


class ModelResource(Resource):
    trained_model: DownloadableResource
    config: Resource

    def __init__(self, trained_model: DownloadableResource, config: Resource):
        super().__init__(trained_model.name, Path())
        self.trained_model = trained_model
        self.config = config


class Resources:
    name: str
    local_root: Path
    resources: List[Resource]

    def __init__(self, name: str, local_root, resources: List[Resource]):
        self.name = name
        self.local_root = Path(local_root)
        self.resources = resources

    def __iter__(self):
        return iter(self.resources)

    def resource_names(self) -> List[str]:
        return [res.name for res in self]

    def get_resource_by_name(self, name: str) -> Resource:
        return next(res for res in self if res.name == name)


def get_non_downloaded_resources(resources: Resources, root_path: Path) -> List[Resource]:
    return [res for res in resources if not (root_path / res.local_path).exists()]


def download_resources(resources: Resources) -> None:
    print(f"Checking {resources.name} resources...")

    root_path = resources.local_root
    if not root_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)

    non_downloaded_resources = get_non_downloaded_resources(resources, root_path)
    if non_downloaded_resources:
        non_downloaded_names = [res.name for res in non_downloaded_resources]
        non_downloaded_files = [cast(DownloadableResource, res).remote_path for res in non_downloaded_resources]

        print(f"Downloading resources: {non_downloaded_names}")

        download(non_downloaded_files, dir=root_path, unzip=True, delete=True)
    else:
        names = [res.name for res in resources]
        print(f"Resources already downloaded: {names}")
