import os
import re
import torch
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS

from svd_ood.utils.data_utils import get_id_labels

def change_by_label(id_dataset: str, dataset: datasets.ImageFolder) -> datasets.ImageFolder:
    """Assign the label of the dataset to the specified label.
    Args:
        id_dataset: the name of the id dataset
        dataset: the object of datasets
    Returns:
        dataset: the dataset of aligning the Label with the specified
    """
    # ID_ImageNet1k is aligned
    if id_dataset == 'ID_ImageNet1K':
        return dataset
    # get the specified label
    labels = get_id_labels(id_dataset)

    # get the name of label
    pattern = "[a-z]+"
    dataset_class = [" ".join(re.findall(pattern, i)) for i in dataset.classes]

    # record the map of dataset label index and specified
    dataset_2_labelIndex = {}
    for idx, label in enumerate(dataset_class):
        dataset_2_labelIndex[dataset.classes[idx]] = labels.index(label)

    # update dataset attributes
    samples = dataset.make_dataset(dataset.root, dataset_2_labelIndex, IMG_EXTENSIONS)
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [s[1] for s in samples]
    dataset.classes = labels
    dataset.class_to_idx = dataset_2_labelIndex

    return dataset


def get_id_loader(data_root, batch_size, id_dataset, split, preprocess, shuffle=False):
    """Set the dataloader for the in-distribution dataset."""
    pin_memory = True if torch.cuda.is_available() else False
    assert split in ["val", "test"], "Invalid split"

    if id_dataset == "ID_ImageNet1K":
        path = os.path.join(data_root, 'ID_ImageNet1K', split)
    elif id_dataset == "ID_COCO":
        path = os.path.join(data_root, 'ID_COCO', split)
    elif id_dataset == "ID_VOC":
        path = os.path.join(data_root, 'ID_VOC', split)
    else:
        raise ValueError(f"Invalid ID dataset: {id_dataset}")

    id_set = datasets.ImageFolder(path, transform=preprocess)
    id_set = change_by_label(id_dataset, id_set)
    id_loader = torch.utils.data.DataLoader(
        id_set, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, pin_memory=pin_memory)
    return id_loader


def get_ood_loader(data_root, batch_size, ood_dataset, split, preprocess):
    """Set the dataloader for the out-of-distribution dataset."""
    pin_memory = True if torch.cuda.is_available() else False
    assert split == "test", "OOD datasets are only available for the test split."

    if ood_dataset == "OOD_iNaturalist":
        path = os.path.join(data_root, 'OOD_iNaturalist', split)
    elif ood_dataset == "OOD_Sun":
        path = os.path.join(data_root, 'OOD_Sun', split)
    elif ood_dataset == "OOD_Places":
        path = os.path.join(data_root, 'OOD_Places', split)
    elif ood_dataset == "OOD_Texture":
        path = os.path.join(data_root, 'OOD_Texture', split)
    elif ood_dataset == "OOD_ImageNet22K":
        path = os.path.join(data_root, 'OOD_ImageNet22K', split)
    elif ood_dataset == "OOD_VOC":
        path = os.path.join(data_root, 'OOD_VOC', split)
    elif ood_dataset == "OOD_COCO":
        path = os.path.join(data_root, 'OOD_COCO', split)
    else:
        raise ValueError(f"Invalid OOD dataset: {ood_dataset}")

    ood_set = datasets.ImageFolder(path, transform=preprocess)
    ood_loader = torch.utils.data.DataLoader(
        ood_set, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=pin_memory)
    return ood_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataloader test")
    parser.add_argument("--data_root", type=str, default="/data/DATASETS/SVD_OOD",
                        help="Path to the root data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    from svd_ood.utils.data_utils import get_ood_datasets
    preprocess = None
    id_datasets = ["ID_ImageNet1K", "ID_COCO", "ID_VOC"]

    split = "val"
    for id_dataset in id_datasets:
        id_loader = get_id_loader(args.data_root, args.batch_size, id_dataset, split, preprocess)
        print(f"\nID dataset: {id_dataset}, {split} set: {len(id_loader.dataset)} images")

    split = "test"
    for id_dataset in id_datasets:
        ood_datasets = get_ood_datasets(id_dataset)
        id_loader = get_id_loader(args.data_root, args.batch_size, id_dataset, split, preprocess)
        print(f"\nID dataset: {id_dataset}, {split} set: {len(id_loader.dataset)} images")
        for ood_dataset in ood_datasets:
            ood_loader = get_ood_loader(args.data_root, args.batch_size, ood_dataset, split, preprocess)
            print(f"OOD dataset: {ood_dataset}, {split} set: {len(ood_loader.dataset)} images")