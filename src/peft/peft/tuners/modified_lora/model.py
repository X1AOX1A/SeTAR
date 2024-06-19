import logging
import random
import torch
import torch.nn as nn

from peft.tuners.lora import LoraModel
from peft.utils import ModulesToSaveWrapper, _get_submodules
logger = logging.getLogger(__name__)


class ModifiedLoraModel(LoraModel):
    """A LoraModel that supports rank specification and svd init."""

    def __init__(self, model, config, adapter_name="default", verbose=True) -> None:
        self.verbose = verbose
        super().__init__(model, config, adapter_name)

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        """Rerite with support for rank specification for each target module."""
        peft_config = self.peft_config[adapter_name]

        # If target_modules is not dict, use the default inject_adapter method
        if not isinstance(peft_config.target_modules, dict):
            return super().inject_adapter(model, adapter_name)

        # Else, use the modified inject_adapter method
        assert peft_config.r is None, "`LoraConfig.r` should be None when `target_modules` is a dict."
        assert peft_config.layers_to_transform is None, \
            "`LoraConfig.layers_to_transform` should be None when `target_modules` is a dict."
        assert peft_config.layers_pattern is None, \
            "`LoraConfig.layers_pattern` should be None when `target_modules` is a dict."

        self._check_new_adapter_config(peft_config)
        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        # Check for modules_to_save in case
        for key in key_list:
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True

        # Inject adapter
        for key, rank in peft_config.target_modules.items():
            assert key in key_list, f"Module {key} not found in the model."
            assert isinstance(rank, int) and rank >= 0, f"Rank should be a nonnegative integer, got {rank}."
            is_target_modules_in_base_model = True
            if rank == 0:
                if self.verbose:
                    logger.info(f"[{key}] [Rank: {rank}] [Skipped]")
            else:
                self.targeted_module_names.append(key)
                parent, target, target_name = _get_submodules(model, key)
                peft_config.r = rank    # set rank for the module
                self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
                if self.verbose:
                    logger.info(f"[{key}] [Rank: {rank}]")
        peft_config.r = None    # reset rank to None

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def disable_lora_scaling(self, adapter="default"):
        """Disable Lora scaling for the specified adapter."""
        def _disable_lora_scaling(module):
            if hasattr(module, "scaling"):
                # backup the scaling factor
                if not hasattr(module, "scaling_backup"):
                    module.scaling_backup = {}
                module.scaling_backup[adapter] = module.scaling[adapter]
                # set scaling factor to 1
                module.scaling[adapter] = 1

        if not hasattr(self, "lora_scaling_disabled") or not self.lora_scaling_disabled:
            self.apply(_disable_lora_scaling)
            self.lora_scaling_disabled = True
            if self.verbose:
                logger.info(f"Disabled Lora scaling for adapter '{adapter}'.")
        else:
            if self.verbose:
                logger.warning(f"Lora scaling has already been disabled for adapter '{adapter}', skipping.")

    def enable_lora_scaling(self, adapter="default"):
        """Enable Lora scaling for the specified adapter."""
        def _enable_lora_scaling(module):
            if hasattr(module, "scaling"):
                assert hasattr(module, "scaling_backup"), "Scaling factor has not been disabled."
                module.scaling[adapter] = module.scaling_backup[adapter]
                del module.scaling_backup[adapter]

        if hasattr(self, "lora_scaling_disabled") and self.lora_scaling_disabled:
            self.apply(_enable_lora_scaling)
            self.lora_scaling_disabled = False
            if self.verbose:
                logger.info(f"Enabled Lora scaling for adapter '{adapter}'.")
        else:
            if self.verbose:
                logger.warning(f"Lora scaling has already been enabled for adapter '{adapter}', skipping.")

    def svd_init(self, adapter="default", lora_weights="small"):
        """Initialize the base weight and Lora weights using SVD for the specified adapter.
        Args:
            adapter (str, optional): The name of the adapter. Defaults to "default".
            lora_weights (str, optional): The type of weights to initialize. Defaults to "small".
                Choose from "small", "large" or "random".
        More details:
            1. Perform SVD on the base weight matrix.
            2. Split the SVD factors into two parts: one for the base weight and the other for the Lora weights.
            3. Initialize the base weight with the large singular values (or small singular values)
                and the Lora weights with the small singular values (or large singular values).
        """
        def _svd_init(module):
            if hasattr(module, "lora_A"):
                rank = module.r[adapter]
                weight = module.base_layer.weight   # shape:[out, in]

                # u:[out,min] | s:[min] | v:[in,min], min=min(in,out)
                u, s, v = torch.svd(weight)     # large to small

                if lora_weights == "small":
                    idx = s.size(0) - rank
                    # u0:[out,min-r] | s0:[min-r] | v0:[in,min-r]
                    u0, s0, v0 = u[:, :idx], s[:idx], v[:, :idx]    # large for weight
                    # u1:[out,r] | s1:[r] | v1:[in,r]
                    u1, s1, v1 = u[:, idx:], s[idx:], v[:, idx:]    # small for lora_A, lora_B
                elif lora_weights == "large":
                    u0, s0, v0 = u[:, rank:], s[rank:], v[:, rank:] # small for weight
                    u1, s1, v1 = u[:, :rank], s[:rank], v[:, :rank] # large for lora_A, lora_B
                elif lora_weights == "random":
                    idx = torch.randperm(s.size(0))
                    u, s, v = u[:, idx], s[idx], v[:, idx]
                    u0, s0, v0 = u[:, rank:], s[rank:], v[:, rank:] # random for weight
                    u1, s1, v1 = u[:, :rank], s[:rank], v[:, :rank] # random for lora_A, lora_B
                else:
                    raise ValueError(f"Invalid value for `lora_weights`: {lora_weights}." + \
                                      "Choose from ['small', 'large', 'random']")
                weight = nn.Parameter(torch.mm(u0, torch.mm(torch.diag(s0), v0.t())))   # [out, in]
                lora_A = nn.Parameter(torch.diag(torch.sqrt(s1)).mm(v1.t()))            # [r, in]
                lora_B = nn.Parameter(torch.mm(u1, torch.diag(torch.sqrt(s1))))         # [out, r]
                assert module.base_layer.weight.size() == weight.size()
                assert module.lora_A[adapter].weight.size() == lora_A.size()
                assert module.lora_B[adapter].weight.size() == lora_B.size()
                requires_grad_base = module.base_layer.weight.requires_grad
                requires_grad_lora_A = module.lora_A[adapter].weight.requires_grad
                requires_grad_lora_B = module.lora_B[adapter].weight.requires_grad
                module.base_layer.weight = weight
                module.lora_A[adapter].weight = lora_A
                module.lora_B[adapter].weight = lora_B
                module.base_layer.weight.requires_grad = requires_grad_base
                module.lora_A[adapter].weight.requires_grad = requires_grad_lora_A
                module.lora_B[adapter].weight.requires_grad = requires_grad_lora_B

        self.apply(_svd_init)
        if self.verbose:
            logger.info(f"SVD initialized for adapter '{adapter}'.")