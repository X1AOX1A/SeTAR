import time
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional

from svd_ood.metrics import Metrics
from svd_ood.scorers import Scorers
from svd_ood.loss import LoCoOpLoss
from peft import PeftModel, ModifiedLoraConfig


class ModelHub:
    def __init__(self, model_type):
        if model_type == "CLIP":
            self.model_class = CLIP()
        elif model_type == "LoCoOp":
            self.model_class = LoCoOp()
        elif model_type == "SwinTransformerV2":
            self.model_class = SwinTransformerV2()
        else:
            raise NotImplementedError(f"{model_type} is not implemented.")

    def load(self, *args, **kwargs):
        return self.model_class.load(*args, **kwargs)

    def compute_scores(self, *args, **kwargs):
        return self.model_class.compute_scores(*args, **kwargs)

    def compute_metrics_loss(self, *args, **kwargs):
        return self.model_class.compute_metrics_loss(*args, **kwargs)

    def get_layer_num(self, *args, **kwargs):
        return self.model_class.get_layer_num(*args, **kwargs)

    def get_weight_name(self, *args, **kwargs):
        return self.model_class.get_weight_name(*args, **kwargs)

    def to_target_modules(self, *args, **kwargs):
        return self.model_class.to_target_modules(*args, **kwargs)

    def apply_svd_prune(self, *args, **kwargs):
        return self.model_class.apply_svd_prune(*args, **kwargs)


class ModelBase:
    """Model Class Template."""

    def __init__(self):
        """Initiate a model class"""
        pass

    def load(self, model_name, device="cuda"):
        """Load model, preprocess and tokenizer.
        Args:
            model_name (str): name or path to the model
            device (str): "cpu" or "cuda"
        Returns:
            model (torch.nn.Module): target model
            train_preprocess (torchvision.transforms.Compose): image preprocessor for training
            val_preprocess (torchvision.transforms.Compose): image preprocessor for inference
            tokenizer: text tokenizer
        """
        model, train_preprocess, val_preprocess, tokenizer = None, None, None, None
        # return model, train_preprocess, val_preprocess, tokenizer
        raise NotImplementedError("load() is not implemented.")

    def compute_scores(self, model, tokenizer, dataloader, id_labels, scorers, temperature,
                       device, verbose=True):
        """Helper function for `run_test_ood` in `test_ood.py`.
        Compute scores for all data in dataloader.
        Args:
            model (torch.nn.Module): target model
            tokenizer: text tokenizer
            dataloader (torch.utils.data.DataLoader): image dataloader
            id_labels (list[str]): in-domain labels
            scorers (list[str]): list of scorers to compute
            temperature (float): temperature for computing scores
            device (str): "cpu" or "cuda"
            verbose (bool): whether to show progress bar
        Returns:
            scores (dict): keys are scorers, values are lists of scores
        """
        raise NotImplementedError

    def compute_metrics_loss(self, model, tokenizer, id_loader, id_labels, scorers,
                             temperature=100, locoop_lambda=0.25, locoop_top_k=200,
                             recall_level=0.95, use_pred_label=False, verbose=False):
        """Helper function for `Seacher` class in `searchers.py`.
        Args:
            model (torch.nn.Module): target model
            tokenizer: text tokenizer
            id_loader (torch.utils.data.DataLoader): in-domain dataloader
            id_labels (list[str]): in-domain labels
            scorers (list[str]): list of scorers to compute, choose from
                ["mcm_score", "l_mcm_score", "gl_mcm_score",
                 "energy_score", "entropy_score", "var_score"]
            temperature (float): temperature for computing scores, 1 for unscaled, 100 for scaled
            locoop_lambda (float): lambda for locoop loss
            locoop_top_k (int): top_k for locoop loss
            recall_level (list[int]): recall levels for metrics
            use_pred_label (bool): whether to use pseudo label for OOD patch detection
            verbose (bool): whether to show progress bar
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
        raise NotImplementedError

    def get_layer_num(self, model_name):
        """Get the number of layers in the model.
        Args:
            model_name (str): name or path to the model
        Returns:
            layer_num (dict): key: tower_type ('visual' or 'text'), value: number of layers
        """
        # return {"visual": None, "text": None}
        raise NotImplementedError("get_layer_num() is not implemented.")

    def _sanity_check(self, tower_type, weight_type, layer_num=None):
        """Check if the input is valid.
        Args:
            tower_type (str): "visual" or "text"
            weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
            layer_num (int): layer number
        """
        assert tower_type in ["visual", "text"], "tower_type must be either 'visual' or 'text'"
        assert weight_type in ["W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"], \
            "weight_type must be one of the following: 'W_q', 'W_k', 'W_v', 'W_o', " + \
            "'W_up', 'W_down', 'W_p'"
        if weight_type in ["W_q", "W_k", "W_v", "W_o", "W_up", "W_down"]:
            assert layer_num is not None, \
                "layer_num must be provided for weight_type 'W_q', 'W_k', 'W_v', 'W_o', 'W_up', 'W_down'"

    def get_weight_name(self, tower_type, weight_type, layer_num=None):
        """Get the weight name in state_dict.
        Args:
            state_dict (dict(torch.tensor)): model.state_dict()
            tower_type (str): "visual" or "text"
            weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
            layer_num (int): layer number
        Returns:
            weight_name (str): weight name in state_dict
        """
        self._sanity_check(tower_type, weight_type, layer_num)
        raise NotImplementedError("get_weight_name() is not implemented.")

    def to_target_modules(self, model, lora_settings, verbose=True):
        """
        Args:
            model (torch.nn.Module): model to prune
            lora_settings (list[list]): List of low rank settings,
                [[tower_type, weight_type, layer_num, rank or rank_ratio], ...]
                e.g. '[['visual','W_q',11,25],['text','W_k',10,30]]
                tower_type (str): "visual" or "text"
                weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
                layer_num (int): layer number
                rank (float): rank ratio or rank, if rank < 1, it will be converted to rank
        Returns:
            target_modules (dict): keys are the module names and values are corresponding rank. e.g.
                {'vision_model.encoder.layers.0.self_attn.q_proj': 50,
                 'vision_model.encoder.layers.0.self_attn.v_proj': 40}
        """
        target_modules = {}
        state_dict = model.state_dict()
        for tower_type, weight_type, layer_num, rank in lora_settings:
            weight_name = self.get_weight_name(tower_type, weight_type, layer_num)
            if rank < 1:
                # convert rank ratio to rank
                full_rank = min(state_dict[weight_name + ".weight"].shape)
                rank_ratio = rank
                rank = round(full_rank * rank_ratio)
                assert rank != full_rank, f"rank_ratio({rank_ratio}) is too large for {weight_name}"
                if verbose:
                    logging.info(
                        f"[{weight_name}] [rank({rank})=round(full_rank({full_rank})*rank_ratio({rank_ratio}))]")
            target_modules[weight_name] = rank
        if verbose:
            logging.info(f"lora_settings -> target_modules: \n{json.dumps(target_modules, indent=4)}")
        return target_modules

    def apply_svd_prune(self, model, args, verbose=True):
        """Apply SVD pruning to the model weights.
        Args:
            model (torch.nn.Module): model to prune
            args.lora_svd_init_type (str): which weights to prune, choose from ["small", "large", "random"]
            args.lora_settings (list[list]): List of low rank settings,
                [[tower_type, weight_type, layer_num, rank], ...]
                e.g. '[['visual','W_q',11,25],['text','W_k',10,30]]
            args.target_modules (dict): UNUSE IF `lora_settings` SPECIFICED
                keys are the module names and values are corresponding rank. e.g.
                {'vision_model.encoder.layers.0.self_attn.q_proj': 50,
                 'vision_model.encoder.layers.0.self_attn.v_proj': 40}
            args.lora_r (int): UNUSE IF `lora_settings` SPECIFICED OR `target_modules` IS DICT
        Returns:
            model (torch.nn.Module): pruned model
        """
        if args.lora_settings:
            # Convert lora_settings to target_modules
            target_modules = self.to_target_modules(model, args.lora_settings, verbose)
        elif args.target_modules:
            target_modules = args.target_modules
        else:
            logging.warning("Niether `lora_settings` nor `target_modules` is provided, lora_svd_init is skipped.")
            return model

        # 1. Create a LoraModel with the specified rank config
        lora_config = ModifiedLoraConfig(r=args.lora_r, target_modules=target_modules)
        model = PeftModel(model, lora_config, verbose=verbose)
        # 2. Split weights with SVD to base model and lora adapter
        model.svd_init(lora_weights=args.lora_svd_init_type)  # "small", "large", "random"
        # 3. Prune and delete lora adapter, back to original model architecture
        model.unload()
        return model


import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from svd_ood.models.clip import CLIPModel
from transformers import CLIPTokenizer, CLIPConfig
class CLIP(ModelBase):
    """CLIP Model Class."""

    def train_preprocess(self, n_px, interpolation):
        return transforms.Compose([
            # scale follow open_clip, https://github.com/mlfoundations/open_clip/blob/9eaf2424e74a4e34f5041e640e5e69bac5eb41aa/src/open_clip/transform.py#L63
            transforms.RandomResizedCrop(n_px, scale=(0.9, 1), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def val_preprocess(self, n_px, interpolation):
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=interpolation),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def load(self, model_name, interpolation=InterpolationMode.BICUBIC, device="cuda"):
        # Note: MCM paper use BILINEAR for interpolation, but we follow OpenAI to use BICUBIC
        with torch.device(device):  # faster to load model on GPU
            try:
                model = CLIPModel.from_pretrained(model_name, local_files_only=True)
                tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
            except OSError:
                model = CLIPModel.from_pretrained(model_name)
                tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model.enable_local_feat()  # return local features, required for locoop loss and GL-MCM score
        train_processor = self.train_preprocess(
            n_px=model.config.vision_config.image_size, interpolation=interpolation)
        val_processor = self.val_preprocess(
            n_px=model.config.vision_config.image_size, interpolation=interpolation)
        return model, train_processor, val_processor, tokenizer

    def compute_scores(self, model, tokenizer, dataloader, id_labels, scorers, temperature,
                       device, verbose=True):
        start_time = time.time()
        # Initialize new scorers each time since they are stateful
        scorers = Scorers(scorers, temperature)
        with torch.no_grad():
            # compute text features
            texts = tokenizer([f"a photo of a {c}" for c in id_labels],
                              return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**texts)
            # normalized text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            tqdm_object = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader
            for batch_idx, (images, labels) in enumerate(tqdm_object):
                # compute image features
                images = images.to(device)
                global_features, local_features = model.get_image_features(images)
                # normalized image features
                global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
                local_features = local_features / local_features.norm(p=2, dim=-1, keepdim=True)
                # cosine similarity as logits **WITH logit_scale.exp() !!!**
                logit_scale = model.logit_scale.exp()
                logits_global = torch.matmul(global_features, text_features.t()) * logit_scale
                logits_local = torch.matmul(local_features, text_features.t()) * logit_scale
                # compute and record scores
                scorers.cal_scores(logits_global, logits_local) # temp=100
        end_time = time.time()
        t = end_time - start_time
        if verbose:
            logging.info(f"Took {round(t, 2)} s to run.")
        return scorers.get_scores()

    def compute_metrics_loss(self, model, tokenizer, id_loader, id_labels, scorers,
                             temperature=100, locoop_lambda=0.25, locoop_top_k=200,
                             recall_level=0.95, use_pred_label=False, verbose=False):
        assert len(scorers) > 0, "At least one scorer is required."
        metrics_dicts = {scorer: {} for scorer in scorers}              # loss, mcm_score, gl_mcm_score, ...
        cal_loss = True if "loss" in scorers else False                 # loss
        scorers = [scorer for scorer in scorers if scorer != "loss"]    # mcm_score, gl_mcm_score, ...
        if len(scorers) > 0:
            # Initialize metrics
            metrics = Metrics(recall_level)
            # Initialize new scorers each time since they are stateful
            id_scores = Scorers(scorers, temperature)    # temp: 1 for unscaled, 100 for scaled
            ood_scores = Scorers(scorers, temperature)   # temp: 1 for unscaled, 100 for scaled
        if cal_loss:
            # Initialize LoCoOpLoss
            loss_fn = LoCoOpLoss(locoop_lambda, locoop_top_k)
            loss_dict = {"locoop_loss": [], "loss_id": [], "loss_ood": [],
                         "acc": [], "ood_patch_percent": []}

        with torch.no_grad():
            # compute text features
            texts = tokenizer([f"a photo of a {c}" for c in id_labels],
                              return_tensors="pt", padding=True, truncation=True).to(model.device)
            text_features = model.get_text_features(**texts)
            # normalized text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            tqdm_object = tqdm(id_loader, total=len(id_loader)) if verbose else id_loader
            for batch_idx, (images, labels_id) in enumerate(tqdm_object):
                # compute image features
                images = images.to(model.device)
                labels_id = labels_id.to(model.device)
                global_features, local_features = model.get_image_features(images)
                # normalized image features
                global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
                local_features = local_features / local_features.norm(p=2, dim=-1, keepdim=True)
                # *scaled* cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_global = torch.matmul(global_features, text_features.t()) * logit_scale
                logits_local = torch.matmul(local_features, text_features.t()) * logit_scale
                batch_size, height, width, cls = logits_local.shape
                logits_local = logits_local.reshape(batch_size, height * width, cls)
                # predicted label in batch
                labels_pred = logits_global.argmax(-1)

                ### extract OOD patches ####
                # use predicted label for OOD patch extraction or use real label
                labels_ood = labels_pred if use_pred_label else labels_id
                # get the top_k local patch labels, shape: [batch, weight*height, top_k]
                _, topk_indices = torch.topk(logits_local, k=locoop_top_k, dim=2)
                # labels: [batch] -> labels_: [batch, weight*height, top_k]
                labels_ood_ = labels_ood.view(batch_size, 1, 1).expand(-1, height * width, locoop_top_k)
                # whether local patch's top-k labels contain id label (true: ID patch, false: OOD patch)
                is_id_patch = topk_indices.eq(labels_ood_).any(dim=2)      # [batch, weight*height]
                is_ood_patch = ~is_id_patch

                ### compute locoop loss ###
                if cal_loss:
                    loss_id = loss_fn.cal_loss_id(logits_global, labels_id)
                    # reuse precomputed `is_ood_patch`, instead of `labels_ood`
                    loss_ood = loss_fn.cal_loss_ood(logits_local, labels=None, is_ood_patch=is_ood_patch)
                    loss_locoop = loss_id + locoop_lambda * loss_ood
                    acc = (labels_pred==labels_id).float().mean() * 100
                    ood_patch_percent = is_ood_patch.float().mean() * 100
                    loss_ = {"locoop_loss": loss_locoop, "loss_id": loss_id, "loss_ood": loss_ood,
                             "acc": acc, "ood_patch_percent": ood_patch_percent}
                    # repeat the loss to match the batch size (for balance)
                    loss_ = {key: val.repeat(len(images)) for key, val in loss_.items()}
                    loss_dict = {key: loss_dict[key] + loss_[key].tolist() for key in loss_dict.keys()}

                ### compute scores ###
                # Note: We use ood patches in ID image as the OOD images w.r.t. the ID image.
                #       Further, for a specific ood patch, we treat itself as the ood global logit,
                #       and all the ood patches as the ood local logits.
                if len(scorers) > 0:
                    # 1. compute ID scores
                    id_scores.cal_scores(logits_global, logits_local.reshape(batch_size, height, width, cls))
                    # 2. compute OOD scores
                    # for each image in the batch, use the ood patches as the OOD images w.r.t. the ID image
                    for i in range(batch_size):
                        # TODO: add ood patch sampling to prevent too many ood patches
                        # extract ood patches as ood global logits
                        logits_global_ood = logits_local[i][is_ood_patch[i]]   # [ood_patch_num, #cls]
                        ood_patch_num = logits_global_ood.shape[0]
                        if ood_patch_num==0:
                            continue
                        # use all ood global patches as the ood local logits w.r.t. the ood global logit
                        logits_local_ood = logits_global_ood.unsqueeze(0).unsqueeze(2).\
                            repeat(ood_patch_num, 1, 1, 1)  # [ood_patch_num, ood_patch_num, 1, #cls]
                        ood_scores.cal_scores(logits_global_ood, logits_local_ood)

        # compute mean metrics and loss
        if len(scorers) > 0:
            id_scores, ood_scores = id_scores.get_scores(), ood_scores.get_scores()
            for scorer in scorers:
                metrics_ = metrics.compute_metrics(id_scores[scorer], ood_scores[scorer])
                metrics_dicts[scorer] = metrics_
        if cal_loss:
            metrics_dicts["loss"] = {key: np.mean(val) for key, val in loss_dict.items()}
        return metrics_dicts

    def get_weight_name(self, tower_type, weight_type, layer_num=None):
        self._sanity_check(tower_type, weight_type, layer_num)
        visual_model = {
            "W_q": "vision_model.encoder.layers.{}.self_attn.q_proj",
            "W_k": "vision_model.encoder.layers.{}.self_attn.k_proj",
            "W_v": "vision_model.encoder.layers.{}.self_attn.v_proj",
            "W_o": "vision_model.encoder.layers.{}.self_attn.out_proj",
            "W_up": "vision_model.encoder.layers.{}.mlp.fc1",
            "W_down": "vision_model.encoder.layers.{}.mlp.fc2",
            "W_p": "visual_projection",
        }
        text_model = {
            "W_q": "text_model.encoder.layers.{}.self_attn.q_proj",
            "W_k": "text_model.encoder.layers.{}.self_attn.k_proj",
            "W_v": "text_model.encoder.layers.{}.self_attn.v_proj",
            "W_o": "text_model.encoder.layers.{}.self_attn.out_proj",
            "W_up": "text_model.encoder.layers.{}.mlp.fc1",
            "W_down": "text_model.encoder.layers.{}.mlp.fc2",
            "W_p": "text_projection",
        }
        if tower_type == "visual":
            return visual_model[weight_type].format(layer_num)
        else:
            return text_model[weight_type].format(layer_num)

    def get_layer_num(self, model_name):
        try:
            config = CLIPConfig.from_pretrained(model_name, local_files_only=True)
        except OSError:
            config = CLIPConfig.from_pretrained(model_name)
        visual_layer_num = config.vision_config.num_hidden_layers
        text_layer_num = config.text_config.num_hidden_layers
        return {"visual": visual_layer_num, "text": text_layer_num}


from svd_ood.models.locoop.modeling_locoop import LoCoOpModel
class LoCoOp(CLIP):
    """LoCoOp: Local regularized Context Optimization.
    https://arxiv.org/pdf/2306.01293.pdf
    """

    def load(self, model_name, n_ctx, locoop_ckpt=None, interpolation=InterpolationMode.BICUBIC, device="cuda", verbose=True):
        # Note: MCM paper use BILINEAR for interpolation, but we follow OpenAI to use BICUBIC
        # only change for the class of model
        with torch.device(device):  # faster to load model on GPU
            try:
                model = LoCoOpModel.from_pretrained(model_name, local_files_only=True)
                tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
            except OSError:
                model = LoCoOpModel.from_pretrained(model_name)
                tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model.enable_local_feat()  # return local features, required for locoop loss and GL-MCM score
        # register prompter with n_ctx
        if verbose:
            logging.info("Registered prompter with n_ctx = {}".format(n_ctx))
        model.register_prompt(n_ctx)
        # load prompter from checkpoint
        if locoop_ckpt is not None:
            if verbose:
                logging.info(f"Loading LoCoOp prompter from {locoop_ckpt}...")
            model.load_prompter(locoop_ckpt)
        else:
            logging.warning("No LoCoOp prompter loaded. Initialize with random prompt.")

        train_processor = self.train_preprocess(
            n_px=model.config.vision_config.image_size, interpolation=interpolation)
        val_processor = self.val_preprocess(
            n_px=model.config.vision_config.image_size, interpolation=interpolation)
        return model, train_processor, val_processor, tokenizer

    def load_prompt(self, model: LoCoOpModel, n_ctx: int, ctx_init: Optional[torch.Tensor],
                    locoop_ckpt: Optional[str]) -> LoCoOpModel:
        """
        load the soft prompt by the given args
        Args:
            model: a LoCoOpModel
            n_ctx: the length of soft prompt
            ctx_init: if given, we will use it to init the soft prompt
            locoop_ckpt: if given we will load the prompt from this file
                could be downloaded from https://github.com/AtsuMiyai/LoCoOp#pre-trained-models
        Returns:
            model: the LoCoOpModel after loading the soft prompt
        """
        model.register_prompt(n_ctx, ctx_init)
        if locoop_ckpt is not None:
            model.load_prompter(locoop_ckpt)
        return model
    def compute_scores(self, model, tokenizer, dataloader, id_labels, scorers, temperature,
                       device, verbose=True):
        start_time = time.time()
        # Initialize new scorers each time since they are stateful
        scorers = Scorers(scorers, temperature)
        with torch.no_grad():
            # compute text features
            texts = tokenizer([f"{c}." for c in id_labels],
                              return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**texts)
            # normalized text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            tqdm_object = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader
            for batch_idx, (images, labels) in enumerate(tqdm_object):
                # compute image features
                images = images.to(device)
                global_features, local_features = model.get_image_features(images)
                # normalized image features
                global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
                local_features = local_features / local_features.norm(p=2, dim=-1, keepdim=True)
                # cosine similarity as logits **WITH logit_scale.exp() !!!**
                logit_scale = model.logit_scale.exp()
                logits_global = torch.matmul(global_features, text_features.t()) * logit_scale
                logits_local = torch.matmul(local_features, text_features.t()) * logit_scale
                # compute and record scores
                scorers.cal_scores(logits_global, logits_local)  # temp=100
        end_time = time.time()
        t = end_time - start_time
        if verbose:
            logging.info(f"Took {round(t, 2)} s to run.")
        return scorers.get_scores()

    def compute_metrics_loss(self, model, tokenizer, id_loader, id_labels, scorers,
                             temperature=100, locoop_lambda=0.25, locoop_top_k=200,
                             recall_level=0.95, use_pred_label=False, verbose=False):
        assert len(scorers) > 0, "At least one scorer is required."
        metrics_dicts = {scorer: {} for scorer in scorers}  # loss, mcm_score, gl_mcm_score, ...
        cal_loss = True if "loss" in scorers else False  # loss
        scorers = [scorer for scorer in scorers if scorer != "loss"]  # mcm_score, gl_mcm_score, ...
        if len(scorers) > 0:
            # Initialize metrics
            metrics = Metrics(recall_level)
            # Initialize new scorers each time since they are stateful
            id_scores = Scorers(scorers, temperature)  # temp: 1 for unscaled, 100 for scaled
            ood_scores = Scorers(scorers, temperature)  # temp: 1 for unscaled, 100 for scaled
        if cal_loss:
            # Initialize LoCoOpLoss
            loss_fn = LoCoOpLoss(locoop_lambda, locoop_top_k)
            loss_dict = {"locoop_loss": [], "loss_id": [], "loss_ood": [],
                         "acc": [], "ood_patch_percent": []}

        with torch.no_grad():
            # compute text features
            texts = tokenizer([f"{c}." for c in id_labels],
                              return_tensors="pt", padding=True, truncation=True).to(model.device)
            text_features = model.get_text_features(**texts)
            # normalized text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            tqdm_object = tqdm(id_loader, total=len(id_loader)) if verbose else id_loader
            for batch_idx, (images, labels_id) in enumerate(tqdm_object):
                # compute image features
                images = images.to(model.device)
                labels_id = labels_id.to(model.device)
                global_features, local_features = model.get_image_features(images)
                # normalized image features
                global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
                local_features = local_features / local_features.norm(p=2, dim=-1, keepdim=True)
                # *scaled* cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_global = torch.matmul(global_features, text_features.t()) * logit_scale
                logits_local = torch.matmul(local_features, text_features.t()) * logit_scale
                batch_size, height, width, cls = logits_local.shape
                logits_local = logits_local.reshape(batch_size, height * width, cls)
                # predicted label in batch
                labels_pred = logits_global.argmax(-1)

                ### extract OOD patches ####
                # use predicted label for OOD patch extraction or use real label
                labels_ood = labels_pred if use_pred_label else labels_id
                # get the top_k local patch labels, shape: [batch, weight*height, top_k]
                _, topk_indices = torch.topk(logits_local, k=locoop_top_k, dim=2)
                # labels: [batch] -> labels_: [batch, weight*height, top_k]
                labels_ood_ = labels_ood.view(batch_size, 1, 1).expand(-1, height * width, locoop_top_k)
                # whether local patch's top-k labels contain id label (true: ID patch, false: OOD patch)
                is_id_patch = topk_indices.eq(labels_ood_).any(dim=2)  # [batch, weight*height]
                is_ood_patch = ~is_id_patch

                ### compute locoop loss ###
                if cal_loss:
                    loss_id = loss_fn.cal_loss_id(logits_global, labels_id)
                    # reuse precomputed `is_ood_patch`, instead of `labels_ood`
                    loss_ood = loss_fn.cal_loss_ood(logits_local, labels=None, is_ood_patch=is_ood_patch)
                    loss_locoop = loss_id + locoop_lambda * loss_ood
                    acc = (labels_pred == labels_id).float().mean() * 100
                    ood_patch_percent = is_ood_patch.float().mean() * 100
                    loss_ = {"locoop_loss": loss_locoop, "loss_id": loss_id, "loss_ood": loss_ood,
                             "acc": acc, "ood_patch_percent": ood_patch_percent}
                    # repeat the loss to match the batch size (for balance)
                    loss_ = {key: val.repeat(len(images)) for key, val in loss_.items()}
                    loss_dict = {key: loss_dict[key] + loss_[key].tolist() for key in loss_dict.keys()}

                ### compute scores ###
                # Note: We use ood patches in ID image as the OOD images w.r.t. the ID image.
                #       Further, for a specific ood patch, we treat itself as the ood global logit,
                #       and all the ood patches as the ood local logits.
                if len(scorers) > 0:
                    # 1. compute ID scores
                    id_scores.cal_scores(logits_global, logits_local.reshape(batch_size, height, width, cls))
                    # 2. compute OOD scores
                    # for each image in the batch, use the ood patches as the OOD images w.r.t. the ID image
                    for i in range(batch_size):
                        # TODO: add ood patch sampling to prevent too many ood patches
                        # extract ood patches as ood global logits
                        logits_global_ood = logits_local[i][is_ood_patch[i]]  # [ood_patch_num, #cls]
                        ood_patch_num = logits_global_ood.shape[0]
                        if ood_patch_num == 0:
                            continue
                        # use all ood global patches as the ood local logits w.r.t. the ood global logit
                        logits_local_ood = logits_global_ood.unsqueeze(0).unsqueeze(2). \
                            repeat(ood_patch_num, 1, 1, 1)  # [ood_patch_num, ood_patch_num, 1, #cls]
                        ood_scores.cal_scores(logits_global_ood, logits_local_ood)

        # compute mean metrics and loss
        if len(scorers) > 0:
            id_scores, ood_scores = id_scores.get_scores(), ood_scores.get_scores()
            for scorer in scorers:
                metrics_ = metrics.compute_metrics(id_scores[scorer], ood_scores[scorer])
                metrics_dicts[scorer] = metrics_
        if cal_loss:
            metrics_dicts["loss"] = {key: np.mean(val) for key, val in loss_dict.items()}
        return metrics_dicts


import math
from transformers import Swinv2Config
from svd_ood.models.swinv2 import Swinv2ForImageClassification
class SwinTransformerV2(ModelBase):
    """SwinTransformerV2 with local featature.
    https://arxiv.org/pdf/2111.09883.pdf
    https://arxiv.org/pdf/2306.01293.pdf
    """

    def train_preprocess(self, n_px, interpolation=InterpolationMode.BICUBIC):
        """Copied from timm
            ```python
            from timm import create_model as create_swin_model
            from timm.data import resolve_model_data_config, create_transform
            model_name == "swinv2_base_window16_256.ms_in1k"
            model = create_swin_model(model_name, pretrained=True)
            data_config = resolve_model_data_config(model)
            preprocess = create_transform(**data_config, is_training=True)
            ```
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(n_px, scale=(0.08, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4850, 0.4560, 0.4060),
                                std=(0.2290, 0.2240, 0.2250))
        ])

    def val_preprocess(self, n_px, interpolation=InterpolationMode.BICUBIC, crop_pct=0.9):
        """Copied from timm
            ```python
            from timm import create_model as create_swin_model
            from timm.data import resolve_model_data_config, create_transform
            model_name == "swinv2_base_window16_256.ms_in1k"
            model = create_swin_model(model_name, pretrained=True)
            data_config = resolve_model_data_config(model)
            preprocess = create_transform(**data_config, is_training=False)
            ```
        """
        scale_size = math.floor(n_px / crop_pct)
        return transforms.Compose([
            transforms.Resize(scale_size, interpolation=interpolation),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4850, 0.4560, 0.4060),
                                std=(0.2290, 0.2240, 0.2250))
        ])

    def load(self, model_name, interpolation=InterpolationMode.BICUBIC, device="cuda"):
        with torch.device(device):  # faster to load model on GPU
            try:
                model = Swinv2ForImageClassification.from_pretrained(model_name, local_files_only=True)
            except OSError:
                model = Swinv2ForImageClassification.from_pretrained(model_name)
        model.enable_local_feat()  # return local features, required for locoop loss and GL-MCM score
        train_processor = self.train_preprocess(
            n_px=model.config.image_size, interpolation=interpolation)
        val_processor = self.val_preprocess(
            n_px=model.config.image_size, interpolation=interpolation)
        tokenizer = None  # Swin Transformer does not have a text tower, so no tokenizer
        return model, train_processor, val_processor, tokenizer

    def compute_scores(self, model, tokenizer, dataloader, id_labels, scorers, temperature,
                       device, verbose=True):
        start_time = time.time()
        # Initialize new scorers each time since they are stateful
        scorers = Scorers(scorers, temperature)
        with torch.no_grad():
            tqdm_object = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader
            for batch_idx, (images, labels) in enumerate(tqdm_object):
                # compute image features
                images = images.to(device)
                outputs = model(images)
                logits_global, logits_local = outputs.logits, outputs.logits_local
                # compute and record scores
                scorers.cal_scores(logits_global, logits_local)
        end_time = time.time()
        t = end_time - start_time
        if verbose:
            logging.info(f"Took {round(t, 2)} s to run.")
        return scorers.get_scores()

    def compute_metrics_loss(self, model, tokenizer, id_loader, id_labels, scorers,
                             temperature=100, locoop_lambda=0.25, locoop_top_k=200,
                             recall_level=0.95, use_pred_label=False, verbose=False):
        assert len(scorers) > 0, "At least one scorer is required."
        metrics_dicts = {scorer: {} for scorer in scorers}              # loss, mcm_score, gl_mcm_score, ...
        cal_loss = True if "loss" in scorers else False                 # loss
        scorers = [scorer for scorer in scorers if scorer != "loss"]    # mcm_score, gl_mcm_score, ...
        if len(scorers) > 0:
            # Initialize metrics
            metrics = Metrics(recall_level)
            # Initialize new scorers each time since they are stateful
            id_scores = Scorers(scorers, temperature)    # temp: 1 for unscaled, 100 for scaled
            ood_scores = Scorers(scorers, temperature)   # temp: 1 for unscaled, 100 for scaled
        if cal_loss:
            # Initialize LoCoOpLoss
            loss_fn = LoCoOpLoss(locoop_lambda, locoop_top_k)
            loss_dict = {"locoop_loss": [], "loss_id": [], "loss_ood": [],
                         "acc": [], "ood_patch_percent": []}

        with torch.no_grad():
            tqdm_object = tqdm(id_loader, total=len(id_loader)) if verbose else id_loader
            for batch_idx, (images, labels_id) in enumerate(tqdm_object):
                # compute image features
                images = images.to(model.device)
                labels_id = labels_id.to(model.device)
                outputs = model(images)
                logits_global, logits_local = outputs.logits, outputs.logits_local
                batch_size, height, width, cls = logits_local.shape
                logits_local = logits_local.reshape(batch_size, height * width, cls)
                # predicted label in batch
                labels_pred = logits_global.argmax(-1)

                ### extract OOD patches ####
                # use predicted label for OOD patch extraction or use real label
                labels_ood = labels_pred if use_pred_label else labels_id
                # get the top_k local patch labels, shape: [batch, weight*height, top_k]
                _, topk_indices = torch.topk(logits_local, k=locoop_top_k, dim=2)
                # labels: [batch] -> labels_: [batch, weight*height, top_k]
                labels_ood_ = labels_ood.view(batch_size, 1, 1).expand(-1, height * width, locoop_top_k)
                # whether local patch's top-k labels contain id label (true: ID patch, false: OOD patch)
                is_id_patch = topk_indices.eq(labels_ood_).any(dim=2)      # [batch, weight*height]
                is_ood_patch = ~is_id_patch

                ### compute locoop loss ###
                if cal_loss:
                    loss_id = loss_fn.cal_loss_id(logits_global, labels_id)
                    # reuse precomputed `is_ood_patch`, instead of `labels_ood`
                    loss_ood = loss_fn.cal_loss_ood(logits_local, labels=None, is_ood_patch=is_ood_patch)
                    loss_locoop = loss_id + locoop_lambda * loss_ood
                    acc = (labels_pred==labels_id).float().mean() * 100
                    ood_patch_percent = is_ood_patch.float().mean() * 100
                    loss_ = {"locoop_loss": loss_locoop, "loss_id": loss_id, "loss_ood": loss_ood,
                             "acc": acc, "ood_patch_percent": ood_patch_percent}
                    # repeat the loss to match the batch size (for balance)
                    loss_ = {key: val.repeat(len(images)) for key, val in loss_.items()}
                    loss_dict = {key: loss_dict[key] + loss_[key].tolist() for key in loss_dict.keys()}

                ### compute scores ###
                # Note: We use ood patches in ID image as the OOD images w.r.t. the ID image.
                #       Further, for a specific ood patch, we treat itself as the ood global logit,
                #       and all the ood patches as the ood local logits.
                if len(scorers) > 0:
                    # 1. compute ID scores
                    id_scores.cal_scores(logits_global, logits_local.reshape(batch_size, height, width, cls))
                    # 2. compute OOD scores
                    # for each image in the batch, use the ood patches as the OOD images w.r.t. the ID image
                    for i in range(batch_size):
                        # TODO: add ood patch sampling to prevent too many ood patches
                        # extract ood patches as ood global logits
                        logits_global_ood = logits_local[i][is_ood_patch[i]]   # [ood_patch_num, #cls]
                        ood_patch_num = logits_global_ood.shape[0]
                        if ood_patch_num==0:
                            continue
                        # use all ood global patches as the ood local logits w.r.t. the ood global logit
                        logits_local_ood = logits_global_ood.unsqueeze(0).unsqueeze(2).\
                            repeat(ood_patch_num, 1, 1, 1)  # [ood_patch_num, ood_patch_num, 1, #cls]
                        ood_scores.cal_scores(logits_global_ood, logits_local_ood)

        # compute mean metrics and loss
        if len(scorers) > 0:
            id_scores, ood_scores = id_scores.get_scores(), ood_scores.get_scores()
            for scorer in scorers:
                metrics_ = metrics.compute_metrics(id_scores[scorer], ood_scores[scorer])
                metrics_dicts[scorer] = metrics_
        if cal_loss:
            metrics_dicts["loss"] = {key: np.mean(val) for key, val in loss_dict.items()}
        return metrics_dicts

    def get_weight_name(self, tower_type, weight_type, layer_num, block_num):
        self._sanity_check(tower_type, weight_type, layer_num)
        visual_model = {
            "W_q": "swinv2.encoder.layers.{}.blocks.{}.attention.self.query",
            "W_k": "swinv2.encoder.layers.{}.blocks.{}.attention.self.key",
            "W_v": "swinv2.encoder.layers.{}.blocks.{}.attention.self.value",
            "W_o": "swinv2.encoder.layers.{}.blocks.{}.attention.output.dense",
            "W_up": "swinv2.encoder.layers.{}.blocks.{}.intermediate.dense",
            "W_down": "swinv2.encoder.layers.{}.blocks.{}.output.dense",
        }
        return visual_model[weight_type].format(layer_num, block_num)

    def to_target_modules(self, model, lora_settings, verbose=True):
        """
        Args:
            model (torch.nn.Module): model to prune
            lora_settings (list[list]): List of low rank settings,
                [[tower_type, weight_type, [layer_num, block_num], rank or rank_ratio], ...]
                e.g. '[['visual','W_q',11,25],['text','W_k',10,30]]
                tower_type (str): "visual" or "text"
                weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
                layer_num (int): layer number
                block_num (int): layer number
                rank (float): rank ratio or rank, if rank < 1, it will be converted to rank
        Returns:
            target_modules (dict): keys are the module names and values are corresponding rank. e.g.
                {'layers.0.blocks.0.intermediate.dense.weight': 50,
                 layers.0.blocks.1.intermediate.dense.weight': 40}
        """
        target_modules = {}
        state_dict = model.state_dict()
        for tower_type, weight_type, num, rank in lora_settings:
            layer_num, block_num = num
            weight_name = self.get_weight_name(tower_type, weight_type, layer_num, block_num)
            if rank < 1:
                # convert rank ratio to rank
                full_rank = min(state_dict[weight_name + ".weight"].shape)
                rank_ratio = rank
                rank = round(full_rank * rank_ratio)
                assert rank != full_rank, f"rank_ratio({rank_ratio}) is too large for {weight_name}"
                if verbose:
                    logging.info(
                        f"[{weight_name}] [rank({rank})=round(full_rank({full_rank})*rank_ratio({rank_ratio}))]")
            target_modules[weight_name] = rank
        if verbose:
            logging.info(f"lora_settings -> target_modules: \n{json.dumps(target_modules, indent=4)}")
        return target_modules

    def get_layer_num(self, model_name):
        try:
            config = Swinv2Config.from_pretrained(model_name, local_files_only=True)
        except OSError:
            config = Swinv2Config.from_pretrained(model_name)
        visual_layer_block_num = config.depths  # list(#num_block)
        return {"visual": visual_layer_block_num}


if __name__ == "__main__":
    import torch
    from PIL import Image
    import requests
    from svd_ood.utils.logger import setup_logger
    setup_logger(log_file=None)

    def clip_test(model_name, device="cuda:0"):
        print(f"Testing CLIP: {model_name}")
        model_hub = ModelHub(model_type="CLIP")
        model, train_preprocess, val_preprocess, tokenizer = model_hub.load(model_name)
        model.to(device)
        model.eval()  # turn off dropout

        layer_num = model_hub.get_layer_num(model_name)
        print("Layer num:", layer_num)

        image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
        print("Downloading image...")

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image = val_preprocess(image).unsqueeze(0).to(device)
        print("image:", list(image.shape))
        text = tokenizer(["a diagram", "a dog", "a cat"], return_tensors="pt").to(device)

        with torch.no_grad():
            # get features from each tower after projection
            print("Getting features from each tower after projection...")
            global_image_features, local_image_features = model.get_image_features(image)
            text_features = model.get_text_features(**text)
            print("global_image_features:", list(global_image_features.shape))  # (images, output_dim)
            print("local_image_features:", list(local_image_features.shape))  # (images, weight, height, texts)
            print("text_features:", list(text_features.shape))  # (texts, output_dim)

            # get cosine similarity (with logit_scale, not softmax)
            print("Getting cosine similarity as logits (with logit_scale, not softmax)...")
            outputs = model(
                pixel_values=image,
                input_ids=text.input_ids,
                attention_mask=text.attention_mask
            )
            image_logits_global = outputs.logits_per_image
            image_logits_local = outputs.logits_per_image_local
            text_logits = outputs.logits_per_text
            print("image_logits_global:", list(image_logits_global.shape))  # (images, texts)
            print("image_logits_local:", list(image_logits_local.shape))  # (images, weight, height, texts)
            print("text_logits:", list(text_logits.shape))  # (texts, images)

            probs = image_logits_global.softmax(dim=-1).cpu().numpy()
            print("Label probs:", probs)


    def locoop_test(model_name, locoop_ckpt: str, n_ctx: int, device="cuda:0"):
        """
        Args:
            model_name: the model name of clip
            locoop_ckpt: the path of the soft prompt checkpoint
                could be downloaded from https://github.com/AtsuMiyai/LoCoOp#pre-trained-models
            n_ctx: the length of soft prompt in locoop model
            device: the GPU index
        """
        print(f"Testing LoCoOp: {model_name}")
        model_hub = ModelHub(model_type="LoCoOp")
        model, train_preprocess, val_preprocess, tokenizer = model_hub.load(model_name, locoop_ckpt=locoop_ckpt,  n_ctx=n_ctx, device=device)
        model.to(device)
        model.eval()  # turn off dropout

        layer_num = model_hub.get_layer_num(model_name)
        print("Layer num:", layer_num)

        image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
        print("Downloading image...")
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image = val_preprocess(image).unsqueeze(0).to(device)
        print("image:", list(image.shape))
        text = tokenizer(["a diagram.", "a dog.", "a cat."], return_tensors="pt").to(device)

        with torch.no_grad():
            # get features from each tower after projection
            print("Getting features from each tower after projection...")
            global_image_features, local_image_features = model.get_image_features(image)
            text_features = model.get_text_features(**text)
            print("global_image_features:", list(global_image_features.shape))  # (images, output_dim)
            print("local_image_features:", list(local_image_features.shape))    # (images, weight, height, texts)
            print("text_features:", list(text_features.shape))                  # (texts, output_dim)

            # get cosine similarity (with logit_scale, not softmax)
            print("Getting cosine similarity as logits (with logit_scale, not softmax)...")
            outputs = model(
                pixel_values=image,
                input_ids=text.input_ids,
                attention_mask=text.attention_mask
            )
            image_logits_global = outputs.logits_per_image
            image_logits_local = outputs.logits_per_image_local
            text_logits = outputs.logits_per_text
            print("image_logits_global:", list(image_logits_global.shape))  # (images, texts)
            print("image_logits_local:", list(image_logits_local.shape))    # (images, weight, height, texts)
            print("text_logits:", list(text_logits.shape))                  # (texts, images)

            probs = image_logits_global.softmax(dim=-1).cpu().numpy()
            print("Label probs:", probs)


    def swinv2_test(model_name, device="cuda:0"):
        print(f"Testing SwinTransformerV2: {model_name}")
        model_hub = ModelHub(model_type="SwinTransformerV2")
        model, train_preprocess, val_preprocess, tokenizer = model_hub.load(model_name)
        model.to(device)
        model.eval()  # turn off dropout

        layer_num = model_hub.get_layer_num(model_name)
        print("Block num at each layer:", layer_num)

        image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
        print("Downloading image...")

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image = val_preprocess(image).unsqueeze(0).to(device)
        print("image:", list(image.shape))

        with torch.no_grad():
            outputs = model(image)
            image_logits_global = outputs.logits
            image_logits_local = outputs.logits_local
            print("image_logits_global:", list(image_logits_global.shape))  # (images, output_dim)
            print("image_logits_local:", list(image_logits_local.shape))    # (images, weight, height, cls_num)

            probs = image_logits_global.softmax(dim=-1).cpu()
            top3_probabilities, top3_class_indices = torch.topk(probs, k=3)
            print("Top3 label probs:", top3_probabilities)
            print("Top3 label index:", top3_class_indices)


    clip_test("openai/clip-vit-base-patch16")
    # Testing CLIP: openai/clip-vit-base-patch16
    # Layer num: {'visual': 12, 'text': 12}
    # Downloading image...
    # image: [1, 3, 224, 224]
    # Getting features from each tower after projection...
    # global_image_features: [1, 512]
    # local_image_features: [1, 14, 14, 512]
    # text_features: [3, 512]
    # Getting cosine similarity as logits (with logit_scale, not softmax)...
    # image_logits_global: [1, 3]
    # image_logits_local: [1, 14, 14, 3]
    # text_logits: [3, 1]
    # Label probs: [[0.7902602  0.19915883 0.01058101]]

    locoop_test("openai/clip-vit-base-patch16",
                locoop_ckpt="/data/MODELS/LoCoOp/checkpoints/seed2/prompt_learner/model.pth.tar-50",
                n_ctx=16)
    # Testing LoCoOp: openai/clip-vit-base-patch16
    # Layer num: {'visual': 12, 'text': 12}
    # Downloading image...
    # image: [1, 3, 224, 224]
    # Getting features from each tower after projection...
    # global_image_features: [1, 512]
    # local_image_features: [1, 14, 14, 512]
    # text_features: [3, 512]
    # Getting cosine similarity as logits (with logit_scale, not softmax)...
    # image_logits_global: [1, 3]
    # image_logits_local: [1, 14, 14, 3]
    # text_logits: [3, 1]
    # Label probs: [[0.88993657 0.10143036 0.00863311]]

    swinv2_test("microsoft/swinv2-base-patch4-window16-256")
    # Testing SwinTransformerV2: microsoft/swinv2-base-patch4-window16-256
    # Block num at each layer: {'visual': [2, 2, 18, 2]}
    # Downloading image...
    # image: [1, 3, 256, 256]
    # image_logits_global: [1, 1000]
    # image_logits_local: [1, 8, 8, 1000]
    # Top3 label probs: tensor([[0.7978, 0.0068, 0.0067]])
    # Top3 label index: tensor([[918, 688, 409]])