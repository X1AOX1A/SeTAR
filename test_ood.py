import time
import logging
import pandas as pd
import torch

from svd_ood.metrics import Metrics
from svd_ood.model_hub import ModelHub
from svd_ood.scorers import save_scores
from svd_ood.utils.utils import setup_seed
from svd_ood.utils.logger import setup_logger
from svd_ood.dataloaders import get_id_loader, get_ood_loader
from svd_ood.utils.data_utils import get_id_labels, get_ood_datasets
from svd_ood.utils.plot_utils import plot_distribution
from svd_ood.utils.argparser import parse_args, print_args, save_args


def run_test_ood(args, verbose=True):
    assert args.split == "test", f"Use 'test' split for OOD detection test, not '{args.split}'"
    if "loss" in args.scorers:
        logging.warning("Loss is not a valid scorer for OOD detection, removing it...")
        args.scorers.remove("loss")
    start_time = time.time()
    setup_seed(args.seed)
    model_hub = ModelHub(args.model_type)

    ### Initialize model, preprocess, tokenizer ###
    # Load the pretrained model
    if verbose:
        logging.info("############ Test OOD Detection ############")
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
            logging.info(f"Loading CLIP model weights from {args.clip_ckpt}...")
        model.load_state_dict(torch.load(args.clip_ckpt, map_location=args.device))

    # Only use the model for evaluation
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    ### Compute scores and metrics ###
    # Initialize metrics
    metrics = Metrics(args.recall_level)
    compute_scores = model_hub.compute_scores

    # Compute ID scores
    id_labels = get_id_labels(args.id_dataset)
    id_loader = get_id_loader(args.data_root, args.batch_size, args.id_dataset, args.split, preprocess)
    if verbose:
        logging.info(f"############ ID dataset: {args.id_dataset}, {args.split} set: "+\
                    f"{len(id_loader.dataset)} images ############")
        logging.info(f"Computing scores for {args.id_dataset}...")
    id_scores = compute_scores(model, tokenizer, id_loader, id_labels,
                               args.scorers, args.temperature, args.device, verbose)
    save_scores(id_scores, args.log_directory, f"scores_{args.id_dataset}_{args.split}.csv")

    # Compute OOD scores
    metrics_dicts = {scorer: {} for scorer in args.scorers}
    cutoffs_dicts = {scorer: {} for scorer in args.scorers}
    ood_datasets = get_ood_datasets(args.id_dataset)
    for ood_dataset in ood_datasets:
        # load ood dataset
        ood_loader = get_ood_loader(args.data_root, args.batch_size, ood_dataset, args.split, preprocess)
        if verbose:
            logging.info(f"############ OOD dataset: {ood_dataset}, {args.split} set: "+\
                        f"{len(ood_loader.dataset)} images ############")
            logging.info(f"Computing scores for {ood_dataset}...")
        # compute scores
        ood_scores = compute_scores(model, tokenizer, ood_loader, id_labels,
                                    args.scorers, args.temperature, args.device, verbose)
        save_scores(ood_scores, args.log_directory, f"scores_{ood_dataset}_{args.split}.csv")
        # compute metrics
        for scorer in args.scorers:
            metrics_ = metrics.compute_metrics(id_scores[scorer], ood_scores[scorer])
            metrics.print_metrics(metrics_, f"{ood_dataset} - {scorer}")
            # record results
            metrics_dicts[scorer][ood_dataset] = metrics_
            cutoffs_dicts[scorer][ood_dataset] = metrics_["cutoff"]
            # plot ID and OOD scores distribution
            plot_distribution(id_scores[scorer], ood_scores[scorer], metrics_["cutoff"],
                              args.log_directory, f"dist_{ood_dataset}_{args.split}_{scorer}.png")

    for scorer in args.scorers:
        if verbose:
            logging.info(f"############ Metrics for {scorer} ############")
        avg_metrics = pd.DataFrame.from_dict(metrics_dicts[scorer], orient='index').mean()
        metrics_dicts[scorer]["Avg"] = avg_metrics.to_dict()
        metrics.save_metrics(metrics_dicts[scorer], args.log_directory, f"metrics_{scorer}_{args.split}.csv")
        metrics.save_cutoffs(cutoffs_dicts[scorer], args.log_directory, f"cutoffs_{scorer}_{args.split}.csv")

    end_time = time.time()
    t = end_time - start_time
    if verbose:
        logging.info(f"############ Done! Test time: {t//60:.0f}m {t%60:.0f}s ############")


if __name__ == "__main__":
    args = parse_args()
    setup_logger(log_dir=args.log_directory, log_file="test_ood.log")
    print_args(args)
    save_args(args, args.log_directory, "config_test_ood.json")
    run_test_ood(args)