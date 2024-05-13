from typing import List, Set


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
        return (f"name: {self.name} \n"
                f"category names: {self.category_names} \n"
                f"super category names: {self.super_category_names}")

    def category_ids(self, categories: CocoCategories) -> List[int]:
        return [category.id for category in categories
                if category.name in self.category_names
                or category.super_category in self.super_category_names]


def filter_image_annotations_internal(image_annotations: dict, categories_filter: CategoriesFilter) -> None:
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
