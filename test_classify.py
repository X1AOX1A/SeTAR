import os
import json
import torch
import logging
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from svd_ood.model_hub import ModelHub
from svd_ood.utils.utils import setup_seed
from svd_ood.utils.logger import setup_logger
from svd_ood.utils.argparser import parse_args, print_args, save_args


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

def save_json(dict_objs, file_name):
    with open(file_name, "w") as f:
        json.dump(dict_objs, f, indent=4)

def set_classify_dataloader(data_root, dataset, preprocess, batch_size):
    """Setup the data loader for the classification task.
    Args:
        data_root: str, the root directory of the dataset.
        dataset: str, the name of the dataset.
            Choose from ["ID_ImageNet1K", "OOD_Sun", "OOD_Places", "OOD_Texture"].
        preprocess: torchvision.transforms, the preprocessing steps.
        batch_size: int, the batch size.
    Returns:
        data_loader: torch.utils.data.DataLoader, the data loader.
        data_labels: list, the list of depulicated labels.
    """
    assert dataset in ["ID_ImageNet1K", "OOD_Sun", "OOD_Places", "OOD_Texture"], \
        "Invalid dataset name."
    image_root = os.path.join(data_root, dataset, "test")
    image_label_file = "./data/{}/{}_test_label.json".format(dataset, dataset.split("_")[1].lower())
    image_label_dict = read_json(image_label_file)
    image_paths = [os.path.join(image_root, img_name) for img_name in image_label_dict.keys()]
    image_labels = list(image_label_dict.values())
    dataset = ClassificationDataset(image_paths, image_labels, transform=preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True if torch.cuda.is_available() else False)
    data_labels = list(set(image_labels))   # depulicated labels
    return data_loader, data_labels

def run_classifications(args, model, preprocess, tokenizer, dataset):
    dataloader, datalabels = set_classify_dataloader(
        args.data_root, dataset, preprocess, args.batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        tqdm_object = tqdm(dataloader, total=len(dataloader))
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.to(args.device)
            texts = tokenizer([f"a photo of a {c}" for c in datalabels], return_tensors="pt",
                              padding=True, truncation=True).to(args.device)
            outputs = model(
                input_ids=texts.input_ids,
                pixel_values=images,
                attention_mask=texts.attention_mask
            )
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)
            pred_label_idxs = probs.argmax(dim=-1).cpu().numpy()
            true_label_idxs = [datalabels.index(label) for label in labels]
            correct += (pred_label_idxs == true_label_idxs).sum()
            total += len(labels)
    acc = correct / total
    return acc


def run_classifications_swin(args, model, preprocess, dataset):
    dataloader, _ = set_classify_dataloader(
        args.data_root, dataset, preprocess, args.batch_size)

    label2id = model.config.label2id
    label2id = {key.split(", ")[0]: val for key, val in label2id.items()}

    correct, total = 0, 0
    with torch.no_grad():
        tqdm_object = tqdm(dataloader, total=len(dataloader))
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.to(args.device)
            outputs = model(images)
            logits_per_image = outputs.logits
            probs = logits_per_image.softmax(dim=-1)
            pred_label_idxs = probs.argmax(dim=-1).cpu().numpy()
            true_label_idxs = [label2id[label] for label in labels]
            correct += (pred_label_idxs == true_label_idxs).sum()
            total += len(labels)
    acc = correct / total
    return acc

def run_test_classify(args, verbose=True):
    setup_seed(args.seed)
    model_hub = ModelHub(args.model_type)

    ### Initialize model, preprocess, tokenizer ###
    # Load the pretrained model
    if verbose:
        logging.info("############ Test Classification ############")
        logging.info(f"Loading {args.model_type} model: {args.model_name}...")
    model_args = {"model_name": args.model_name, "device": args.device}
    if args.model_type == "LoCoOp":
        model_args.update({"n_ctx": args.n_ctx, "locoop_ckpt": args.locoop_ckpt, "verbose": verbose})
    model, _, preprocess, tokenizer = model_hub.load(**model_args)

    # Apply SVD pruning to the model weights
    if args.lora_svd_init:
        if verbose:
            logging.info(f"Applying SVD prune to '{args.lora_svd_init_type}' weights...")
        model = model_hub.apply_svd_prune(model, args, verbose=verbose)

    # Load the weight from checkpoint if specified
    if args.clip_ckpt:
        if verbose:
            logging.info(f"Loading clip model weights from {args.clip_ckpt}...")
        model.load_state_dict(torch.load(args.clip_ckpt, map_location=args.device))

    # Only use the model for evaluation
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    ### Compute accuracy ###
    # run classification on each dataset test set
    acc_dict = {}
    for dataset in ["ID_ImageNet1K", "OOD_Sun", "OOD_Places", "OOD_Texture"]:
        if verbose:
            logging.info(f"############ {dataset} ############")
        if args.model_type == "SwinTransformerV2":
            if dataset != "ID_ImageNet1K":
                continue    # swin ckpt only trained on ImageNet1K
            acc = run_classifications_swin(args, model, preprocess, dataset)
        else:
            acc = run_classifications(args, model, preprocess, tokenizer, dataset)
        acc = acc*100
        acc_dict[dataset] = acc
        if verbose:
            logging.info(f"Accuracy on {dataset}: {acc:.2f}")

    if verbose:
        logging.info(f"############ Summary ############")
        for dataset, acc in acc_dict.items():
            logging.info(f"Accuracy on {dataset}: {acc:.2f}")
    save_json(acc_dict, os.path.join(args.log_directory, f"metrics_classify_{args.split}.json"))


if __name__ == "__main__":
    args = parse_args()
    setup_logger(log_dir=args.log_directory, log_file="test_classfiy.log")
    print_args(args)
    save_args(args, args.log_directory, "config_test_classfiy.json")
    run_test_classify(args)