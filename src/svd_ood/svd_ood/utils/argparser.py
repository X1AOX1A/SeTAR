import os
import ast
import json
import logging
import argparse
from transformers import SchedulerType

def parse_args(search=False, finetune=False):
    """Parse arguments for the experiment.
    Args:
        search (bool): whether to parse arguments for search process
        finetune (bool): whether to parse arguments for finetune process
    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for the experiment")
    parser.add_argument("--config_file", type=str, default=None, help="Path to the config file")

    # Experiment settings
    parser.add_argument("--data_root", type=str, default="./data", help="Path to the data root")
    parser.add_argument("--id_dataset", type=str, default="ID_ImageNet1K",
                        choices=["ID_ImageNet1K", "ID_COCO", "ID_VOC"], help="ID dataset to use")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"], help="Split to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--log_directory", type=str, default="./debug", help="Directory to save logs")

    # Scorer settings
    parser.add_argument("--scorers", type=str, nargs="+", default=["mcm_score"],
                        choices=["loss", "mcm_score", "l_mcm_score", "gl_mcm_score",
                                 "energy_score", "entropy_score", "var_score"],
                        help="Scorers to use. Set 'loss' if you want to use 'locoop_loss' " + \
                             "as best_metric for searching ('loss' only apply to search).")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for output scores")
    parser.add_argument("--recall_level", type=float, default=0.95,
                        help="Recall level for ID dataset, used for FPR metric, e.g. FPR95")

    # Model settings
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="Model name or path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="CLIP",
                        choices=["CLIP", "LoCoOp", "SwinTransformerV2"], help="Model type")
    parser.add_argument("--clip_ckpt", type=str, default=None,
                        help="Path to CLIP checkpoint, set for finetuned model. " + \
                             "e.g. path/to/clip/model.pth.tar-50")
    parser.add_argument("--locoop_ckpt", type=str, default=None,
                        help="Path to locoop checkpoint, set if model_type is locoop")

    # SVD low rank settings
    parser.add_argument("--lora_svd_init", type=bool, default=True,
                        help="Whether to apply SVD LoRA initialization (for finetune) or SVD pruning (for search)")
    parser.add_argument("--lora_svd_init_type", type=str, default=None,
                        choices=["small", "large", "random"], help="Weights for SVD LoRA initialization or pruning")
    parser.add_argument("--lora_settings", type=list_of_lists_type, default=None,
                        help="List of low rank settings, will be converted to target_modules "+
                        "[[tower_type, weight_type, layer_num, rank or rank_ratio], ...]" + \
                        "if rank < 1, it will be converted to rank" + \
                        "e.g. '[['visual','W_q',11,25],['text','W_k',10,30]]'")
    parser.add_argument("--target_modules", type=list_type, default=None,
                        help="""List of module names or regex expression of the module names to replace with LoRA.
                        e.g. ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'.
                        This can also be a dictionary where the keys are the module names and
                        values are corresponding rank. For example,
                        {
                            'vision_model.encoder.layers.0.self_attn.q_proj': 50,
                            'vision_model.encoder.layers.0.self_attn.v_proj': 40
                        }.
                        Notice that you should specify the name of the module exactly as it appears in the model.
                        `r`, `layers_to_transform` and `layers_pattern` will be ignored when `target_modules` is a dict.""")
    parser.add_argument("--lora_r", type=int, default=None,
                        help="Rank for LoRA, ignored if target_modules is a dict")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Alpha for LoRA, notice that scaling is disable for SVD LoRA, thus no need to set for SVD LoRA.")
    parser.add_argument("--n_ctx", type=int, default=16,
                        help="the length of soft prompt int locoop model")

    # Search settings
    if search:
        parser.add_argument("--searcher", type=str, default="visual_text",
                            choices=["visual_text", "test_visual", "text_only", "visual_only", "visual_only_swin"
                                     "layer_exhaustive", "modality_interleaved"],
                            help="Searcher to use")
        parser.add_argument("--candi_ratios", type=float, nargs="+",
                            default=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                            help="List of candidate reduction ratios")
        parser.add_argument("--weight_type", type=str, default="W_up",
                            choices=["W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"],
                            help="Layer type to search")
        parser.add_argument("--best_scorer", type=str, default="mcm_score",
                            choices=["loss", "mcm_score", "l_mcm_score", "gl_mcm_score",
                                     "energy_score", "entropy_score", "var_score"],
                            help="Scorer to use for best setting")
        parser.add_argument("--best_metric", type=str, default="auroc",
                            choices=["locoop_loss", "loss_id", "loss_ood", "auroc", "aupr", "fpr95"],
                            help="Best metric to select the best settings, if best_scorer is 'loss', " + \
                                 "choose from ['locoop_loss', 'loss_id', 'loss_ood'], " + \
                                 "otherwise, choose from ['auroc', 'aupr', 'fpr95']")
        parser.add_argument("--freeze_proj", type=bool, default=True,
                            help="Freeze projection layer during search. " + \
                                 "If False, additonal projection layer(s) will be added")
        parser.add_argument("--locoop_lambda", type=float, default=0.25,
                            help="Lambda value for locoop OOD loss")
        parser.add_argument("--locoop_top_k", type=int, default=200,
                            help="Top k value for LoCoOp OOD loss")
        parser.add_argument("--use_pred_label", type=bool, default=False,
                            help="Whether to use pred label for OOD patches detection")

    # Finetune settings
    if finetune:
        parser.add_argument("--num_train_epochs", type=int, default=3,
            help="Total number of training epochs to perform.")
        parser.add_argument("--max_train_steps", type=int, default=None,
            help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", type=float, default=0.002,
            help="Initial learning rate (after the potential warmup period) to use.")
        parser.add_argument("--weight_decay", type=float, default=0.0,
            help="Weight decay to use.")
        parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum")
        parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
            help="The scheduler type to use.")
        parser.add_argument("--num_warmup_steps", type=int, default=0,
            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--locoop_lambda", type=float, default=0.25,
            help="Lambda value for LoCoOp OOD loss")
        parser.add_argument("--locoop_top_k", type=int, default=200,
            help="Top k value for LoCoOp OOD loss")
        parser.add_argument("--logging_steps", type=int, default=1,
            help="Log every X updates steps.")


    args = parser.parse_args()

    # Load args from config file
    if args.config_file:
        args = load_args_from_config_file(args, args.config_file)

    # Update log directory if it contains args
    if "args." in args.log_directory: # e.g. "./results/{args.id_dataset}/{args.scorers[0]}"
        args.log_directory = args.log_directory.format(args=args)
    return args

def list_type(string):
    if string is None or string.lower() == 'null':
        return None
    try:
        value = ast.literal_eval(string)
        if isinstance(value, list):
            return value
        else:
            raise argparse.ArgumentTypeError("The input is not a list")
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("The input is not a valid list format")

def list_of_lists_type(string):
    if string is None or string.lower() == 'null':
        return None
    try:
        value = ast.literal_eval(string)
        if not all(isinstance(item, list) for item in value):
            raise ValueError
        return value
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Argument should be a list of lists")

def load_args_from_config_file(args, config_file):
    """Overwrite args with the settings in the config file.
    Args:
        args (argparse.Namespace): original arguments
        config_file (str): path to the config file
    Returns:
        args (argparse.Namespace): updated arguments
    """
    with open(config_file, "r") as f:
        file_args = argparse.Namespace(**json.load(f))
        file_args.config_file = config_file
    args_dict = vars(args)
    file_args_dict = vars(file_args)
    for k, v in file_args_dict.items():
        if k in args_dict:
            args_dict[k] = v
        else:
            raise ValueError(f"Invalid key in config file: {k}")
    args = argparse.Namespace(**args_dict)
    return args

def print_args(args):
    if args.config_file:
        logging.info(f"Loading args from config file: {args.config_file}")
    msg = {k: v for k, v in args.__dict__.items() if k!="config_file"}
    msg = f"Config: \n" + json.dumps(msg, indent=4)
    logging.info(msg)

def save_args(args, file_path, file_name):
    """Save arguments to a JSON file."""
    args_dict = vars(args)
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
    # save all args except args.config_file
    with open(os.path.join(file_path, file_name), "w") as f:
        json.dump({k: v for k, v in args_dict.items() if k!="config_file"}, f, indent=4)

if __name__ == "__main__":
    from svd_ood.utils.logger import setup_logger
    setup_logger(log_file=None)
    args = parse_args()
    print_args(args)
    save_args(args, args.log_directory, "config.json")