import time
import torch
import logging
from tqdm import tqdm

from svd_ood.model_hub import ModelHub
from svd_ood.utils.utils import setup_seed
from svd_ood.dataloaders import get_id_loader
from svd_ood.utils.data_utils import get_id_labels
from svd_ood.utils.search_utils import MetricHelper, ResultHelper, SettingHelper
from svd_ood.models.locoop.modeling_locoop import LoCoOpModel


class Searcher:
    def __init__(self, searcher, *args, **kwargs):
        if searcher == "visual_text":
            self.searcher = VisualTextSearch(*args, **kwargs)
        elif searcher == "text_visual":
            self.searcher = TextVisualSearch(*args, **kwargs)
        elif searcher == "text_only":
            self.searcher = TextSearch(*args, **kwargs)
        elif searcher == "visual_only":
            self.searcher = VisualSearch(*args, **kwargs)
        elif searcher == "visual_only_swin":
            self.searcher = VisualSearchForSwin(*args, **kwargs)
        elif searcher == "modality_interleaved":
            self.searcher = ModalityInterleavedSearch(*args, **kwargs)
        elif searcher == "layer_exhaustive":
            self.searcher = LayerExhaustiveSearch(*args, **kwargs)
        else:
            raise ValueError(f"searcher {searcher} is not supported.\n" + \
                             f"Supported searchers: {self.support_searchers()}")
        logging.info("############ Searcher Created ############")
        logging.info(f"Searcher: {searcher}")

    def support_searchers(self):
        return ["visual_text", "text_visual", "text_only", "visual_only", "visual_only_swin",
                "modality_interleaved", "layer_exhaustive"]

    def run(self, args):
        return self.searcher.run(args)


class BaseSearch(MetricHelper, ResultHelper, SettingHelper):
    def __init__(self, candi_ratios, layer_num, weight_type, best_metric="auroc",
                 best_scorer="mcm_score", freeze_proj=True):
        """Base class for low rank settings search.
        Args:
            candi_ratios (list[float]): candidate ratios for low rank settings
            layer_num (dict): keys are "visual" and (or) "text", values are the number of layers
            weight_type (str): weight type for low rank searching
            best_metric (str): best metric to select the best settings,
                if best_scorer is "loss" , choose from ["locoop_loss", "loss_id", "loss_ood"],
                otherwise, choose from ["auroc", "aupr", "fpr95"]
            best_scorer (str): best scorer used to select the best settings, choose from
                ["loss", "mcm_score", "l_mcm_score", "gl_mcm_score", "energy_score", "entropy_score", "var_score"]
            freeze_proj (bool): whether to freeze projection matrix, default: True
        """
        super().__init__()
        self.candi_ratios = candi_ratios
        self.layer_num = layer_num
        self.weight_type = weight_type
        self.best_scorer = best_scorer
        self.best_metric_name = best_metric
        self.freeze_proj = freeze_proj
        logging.info(f"Best scorer: {best_scorer}")
        logging.info(f"Best metric: {best_metric}")
        if "loss" in best_metric:
            assert best_scorer == "loss", f"best_scorer should be 'loss' if best_metric {best_metric}"

    def run(self, args):
        """Run the search process.
        Args:
            args (argparse.Namespace): arguments
        Returns:
            best_settings (list): best low rank settings,
                [[tower_type, weight_type, layer_num, reduc_ratio], ...]
        """
        best_settings = None
        raise NotImplementedError

    def compute_metrics_loss(self, args, verbose=False):
        """Compute metrics and loss for the given low rank settings.
        Returns:
            metrics_dicts (dict): metrics for the given low rank settings, e.g.:
            {
                "loss": {
                    "locoop_loss": mean locoop_loss (np.array),
                    "loss_id": mean id loss (np.array),
                    "loss_ood": mean ood loss (np.array),
                }
                # 'mcm_score', 'gl_mcm_score', ...
                "scorer_name":{
                    "auroc": auroc (np.array),
                    "aupr": aupr (np.array),
                    "fpr": fpr (np.array),
                    "cutoff": cutoff (np.array),
                }
            }
        """
        assert args.split == "val", f"Use 'val' split for development, not '{args.split}'"
        setup_seed(args.seed)
        model_hub = ModelHub(args.model_type)

        ### Initialize model, preprocess, tokenizer ###
        # Load the pretrained model
        if verbose:
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
        if isinstance(model , LoCoOpModel):
            model.register_prompt(args.n_ctx)
            if args.locoop_ckpt is not None:
                model.load_prompter(args.locoop_ckpt)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # Compute metrics
        id_labels = get_id_labels(args.id_dataset)
        id_loader = get_id_loader(args.data_root, args.batch_size, args.id_dataset, args.split, preprocess)
        metrics_dicts = model_hub.compute_metrics_loss(
            model=model,
            tokenizer=tokenizer,
            id_loader=id_loader,
            id_labels=id_labels,
            scorers=args.scorers,
            temperature=args.temperature,
            locoop_lambda=args.locoop_lambda,
            locoop_top_k=args.locoop_top_k,
            recall_level=args.recall_level,
            use_pred_label=args.use_pred_label,
        )
        return metrics_dicts


class VisualTextSearch(BaseSearch):
    """Search low rank settings for visual and text towers sequentially from top to bottom.
    Time complexity: O(N*M) or 2*N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # visual_11, visual_10, ..., visual_0, text_11, text_10, ..., text_0
        self.tower_weight_layer = \
            ([("visual", "W_p", None)] if not self.freeze_proj else []) + \
            [("visual", self.weight_type, i) for i in range(self.layer_num["visual"]-1, -1, -1)] + \
            ([("text", "W_p", None)] if not self.freeze_proj else []) + \
            [("text", self.weight_type, i) for i in range(self.layer_num["text"]-1, -1, -1)]

    def run(self, args):
        start_time = time.time()
        logging.info(f"Start searching low rank configs...")
        self.init_best_metric(args.best_metric)
        for step, (tower_type, weight_type, layer_num) in tqdm(enumerate(self.tower_weight_layer)):
            # set best ratio to 0 for each step (disable low rank if it get worse)
            best_setting = [tower_type, weight_type, layer_num, 0]
            # loop over candidate ratios
            for tmp_ratio in tqdm(self.candi_ratios,
                                  desc=f"Searching [{tower_type}-{weight_type}-{layer_num}]"):
                # add temporary setting
                tmp_setting = [tower_type, weight_type, layer_num, tmp_ratio]
                self.add_setting(*tmp_setting)
                # update lora_settings for the next run
                args.lora_settings = self.get_settings()
                # metrics_dicts, {"loss": {locoop_loss, loss_id, loss_ood},
                #                 "scorers": {aurouc, aupr, fpr, cutoff}}
                metrics_dicts = self.compute_metrics_loss(args)
                # remove temporary setting
                self.pop_setting()
                # log temporary setting and metrics that computed
                self.add_result(step, metrics_dicts, *tmp_setting)
                # check if the current setting is the best
                # {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
                avg_metrics = metrics_dicts[self.best_scorer]
                if self.is_better(avg_metrics):
                    self.update_best_metrics(avg_metrics)
                    best_setting = [tower_type, weight_type, layer_num, tmp_ratio]
            # add the best metrics and setting
            best_metrics = self.get_best_metrics()
            self.add_setting(*best_setting)
            self.add_best_results(step, best_metrics, *best_setting)
            logging.info(f"Current best settings: \n{self.get_best_results_string()}\n")
            # save best settings and score logs at each step
            self.save_best_results(args.log_directory, "search_best_results.csv")
            self.save_results(args.log_directory, "search_results.json")

        best_settings = self.get_settings()
        logging.info(f"Best low rank configs: \n{best_settings}")
        end_time = time.time()
        t = end_time - start_time
        logging.info(f"############ Done! Search time: {t//60:.0f}m {t%60:.0f}s ############")
        return best_settings


class TextVisualSearch(VisualTextSearch):
    """Search low rank settings for text and visual towers sequentially from top to bottom.
    Time complexity: O(N*M) or 2*N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # text_11, text_10, ..., text_0, visual_11, visual_10, ..., visual_0
        self.tower_weight_layer = \
            ([("text", "W_p", None)] if not self.freeze_proj else []) + \
            [("text", self.weight_type, i) for i in range(self.layer_num["text"]-1, -1, -1)] + \
            ([("visual", "W_p", None)] if not self.freeze_proj else []) + \
            [("visual", self.weight_type, i) for i in range(self.layer_num["visual"]-1, -1, -1)]


class TextSearch(VisualTextSearch):
    """Search low rank settings for text tower from top to bottom.
    Time complexity: O(N*M) or N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # text_11, text_10, ..., text_0
        self.tower_weight_layer = \
            ([("text", "W_p", None)] if not self.freeze_proj else []) + \
            [("text", self.weight_type, i) for i in range(self.layer_num["text"]-1, -1, -1)]


class VisualSearch(VisualTextSearch):
    """Search low rank settings for visual tower from top to bottom.
    Time complexity: O(N*M) or N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # visual_11, visual_10, ..., visual_0
        self.tower_weight_layer = \
            ([("visual", "W_p", None)] if not self.freeze_proj else []) + \
            [("visual", self.weight_type, i) for i in range(self.layer_num["visual"]-1, -1, -1)]


class ModalityInterleavedSearch(VisualTextSearch):
    """Search low rank settings for visual and text towers interleaved from top to bottom.
    Time complexity: O(N*M) or 2*N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.layer_num["visual"] == self.layer_num["text"], \
            f"ModalityInterleavedSearch requires the same number of layers for " + \
            f"visual ({self.layer_num['visual']}) and text ({self.layer_num['text']}) towers."
        # visual_11, text_11, visual_10, text_10, ..., visual_0, text_0
        self.tower_weight_layer = \
            [([("visual", "W_p", None), ("text", "W_p", None)] if not self.freeze_proj else []) + \
            [("visual", self.weight_type, i), ("text", self.weight_type, i)]
             for i in range(self.layer_num["visual"]-1, -1, -1)]
        self.tower_weight_layer = sum(self.tower_weight_layer, [])  # flatten the list


class LayerExhaustiveSearch(BaseSearch):
    """Search low rank settings for visual and text towers exhaustively.
    Time complexity: O(N^2*M) or (N+1)*N*M, where
        N is the number of layers
        M is the number of candidate ratios (len(candi_ratios))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tower_weight_layer = \
            ([("visual", "W_p", None)] if not self.freeze_proj else []) + \
            [("visual", self.weight_type, i) for i in range(self.layer_num["visual"]-1, -1, -1)] + \
            ([("text", "W_p", None)] if not self.freeze_proj else []) + \
            [("text", self.weight_type, i) for i in range(self.layer_num["text"]-1, -1, -1)]

    def run(self, args):
        start_time = time.time()
        logging.info(f"Start searching low rank configs...")
        self.init_best_metric(args.best_metric)
        step, n_step = 0, len(self.tower_weight_layer)-1
        progress_bar = tqdm(total=n_step)

        while len(self.tower_weight_layer) > 0:
            # init best setting with the first tuple and ratio 0
            tower, weight, layer = self.tower_weight_layer[0]
            best_setting = [tower, weight, layer, 0]
            # loop over all tuple that not determined yet
            for tower_type, weight_type, layer_num in tqdm(self.tower_weight_layer):
                # loop over candidate ratios
                for tmp_ratio in tqdm(self.candi_ratios,
                                      desc=f"Searching [{tower_type}-{weight_type}-{layer_num}]"):
                    # add temporary setting
                    tmp_setting = [tower_type, weight_type, layer_num, tmp_ratio]
                    self.add_setting(*tmp_setting)
                    # update lora_settings for the next run
                    args.lora_settings = self.get_settings()
                    # metrics_dicts, {"loss": {locoop_loss, loss_id, loss_ood},
                    #                 "scorers": {aurouc, aupr, fpr, cutoff}}
                    metrics_dicts = self.compute_metrics_loss(args)
                    # remove temporary setting
                    self.pop_setting()
                    # log temporary setting and metrics that computed
                    self.add_result(step, metrics_dicts, *tmp_setting)
                    # check if the current setting is the best
                    # {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
                    avg_metrics = metrics_dicts[self.best_scorer]
                    if self.is_better(avg_metrics):
                        self.update_best_metrics(avg_metrics)
                        best_setting = [tower_type, weight_type, layer_num, tmp_ratio]
            # add the best metrics and setting
            best_metrics = self.get_best_metrics()
            self.add_setting(*best_setting)
            self.add_best_results(step, best_metrics, *best_setting)
            logging.info(f"[Step: {step}/{n_step}] "+\
                             f"Current best settings: \n{self.get_best_results_string()}\n")
            # remove the tuple that have been determined
            tower, weight, layer, _ = best_setting
            self.tower_weight_layer.remove((tower, weight, layer))
            # save best settings and score logs at each step
            self.save_best_results(args.log_directory, "search_best_results.csv")
            self.save_results(args.log_directory, "search_results.json")
            step += 1
            progress_bar.update(1)

        best_settings = self.get_settings()
        logging.info(f"Best low rank configs: \n{best_settings}")
        end_time = time.time()
        t = end_time - start_time
        logging.info(f"############ Done! Search time: {t//60:.0f}m {t%60:.0f}s ############")
        return best_settings


class VisualSearchForSwin(BaseSearch):
    """Specialized VisualSearch for SwinTransformer.
    SwinTransformer involves multiple layers and blocks, thus diff from other models.
    Also notice that SwinTransformer only has visual tower, thus only VisualSearch is implemented.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # visual_3_1, visual_3_0, visual_2_17, ..., visual_2_0,
        # visual_1_1, visual_1_0, visual_0_1, visual_0_0
        layer_num_list = list(range(len(self.layer_num["visual"])))
        block_num_list = self.layer_num["visual"]
        self.tower_weight_layer = [("visual", "W_p", (None, None))] if not self.freeze_proj else []
        for layer_num, block_num in zip(layer_num_list[::-1], block_num_list[::-1]):
            self.tower_weight_layer += [
                ("visual", self.weight_type, (layer_num, block)) for block in range(block_num-1, -1, -1)]

    def run(self, args):
        start_time = time.time()
        logging.info(f"Start searching low rank configs...")
        self.init_best_metric(args.best_metric)
        for step, (tower_type, weight_type, layer_block) in tqdm(enumerate(self.tower_weight_layer)):
            # set best ratio to 0 for each step (disable low rank if it get worse)
            best_setting = [tower_type, weight_type, layer_block, 0]
            # loop over candidate ratios
            for tmp_ratio in tqdm(self.candi_ratios,
                                  desc=f"Searching [{tower_type}-{weight_type}-{layer_block}]"):
                # add temporary setting
                tmp_setting = [tower_type, weight_type, layer_block, tmp_ratio]
                self.add_setting(*tmp_setting)
                # update lora_settings for the next run
                args.lora_settings = self.get_settings()
                # metrics_dicts, {"loss": {locoop_loss, loss_id, loss_ood},
                #                 "scorers": {aurouc, aupr, fpr, cutoff}}
                metrics_dicts = self.compute_metrics_loss(args)
                # remove temporary setting
                self.pop_setting()
                # log temporary setting and metrics that computed
                self.add_result(step, metrics_dicts, *tmp_setting)
                # check if the current setting is the best
                # {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
                avg_metrics = metrics_dicts[self.best_scorer]
                if self.is_better(avg_metrics):
                    self.update_best_metrics(avg_metrics)
                    best_setting = [tower_type, weight_type, layer_block, tmp_ratio]
            # add the best metrics and setting
            best_metrics = self.get_best_metrics()
            self.add_setting(*best_setting)
            self.add_best_results(step, best_metrics, *best_setting)
            logging.info(f"Current best settings: \n{self.get_best_results_string()}\n")
            # save best settings and score logs at each step
            self.save_best_results(args.log_directory, "search_best_results.csv")
            self.save_results(args.log_directory, "search_results.json")

        best_settings = self.get_settings()
        logging.info(f"Best low rank configs: \n{best_settings}")
        end_time = time.time()
        t = end_time - start_time
        logging.info(f"############ Done! Search time: {t//60:.0f}m {t%60:.0f}s ############")
        return best_settings