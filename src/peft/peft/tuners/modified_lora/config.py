import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType
logger = logging.getLogger(__name__)


@dataclass
class ModifiedLoraConfig(LoraConfig):
    """A LoraConfig that supports rank specification for each target module."""

    r: Optional[int] = field(
        default=None,
        metadata={
            "help": "Rank to use for all target modules. If `target_modules` is a dict, this will be ignored."
        },
    )

    target_modules: Optional[Union[list[str], str, dict]] = field(
        default=None,
        metadata={
            "help": (
                """
                List of module names or regex expression of the module names to replace with LoRA.
                For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'.
                This can also be a dictionary where the keys are the module names to replace and 
                the values are corresponding rank. For example, 
                {
                    'vision_model.encoder.layers.0.self_attn.q_proj': 50, 
                    'vision_model.encoder.layers.0.self_attn.v_proj': 40
                }. 
                Notice that you should specify the name of the module exactly as it appears in the model. 
                `r`, `layers_to_transform` and `layers_pattern` will be ignored when `target_modules` is a dict.
                """
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MODIFIED_LORA

        # Sanity checks when target_modules is a dict
        if isinstance(self.target_modules, dict):
            if self.r is not None:
                logger.warning("`LoraConfig.r` will be ignored since `target_modules` is a dict.")      
                logger.warning("Rank will be set based on the values in `target_modules`.")
                logger.warning("`LoraConfig.r` is set to None.")
                self.r = None
            if self.layers_to_transform is not None:
                logger.warning("`LoraConfig.layers_to_transform` will be ignored since `target_modules` is a dict.")
                logger.warning("Layers to transform will be set based on the keys in `target_modules`.")
                logger.warning("`LoraConfig.layers_to_transform` is set to None.")
                self.layers_to_transform = None
            if self.layers_pattern is not None:
                logger.warning("`LoraConfig.layers_pattern` will be ignored since `target_modules` is a dict.")
                logger.warning("Layers pattern will be set based on the keys in `target_modules`.")
                logger.warning("`LoraConfig.layers_pattern` is set to None.")
                self.layers_pattern = None

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.use_dora and self.megatron_config:
            raise ValueError("DoRA does not support megatron_core, please set `use_dora=False`.")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            import importlib

            if not importlib.util.find_spec("scipy"):
                raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
            if self.loftq_config is None:
                raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")

        # convert loftq_config to dict
        if self.loftq_config and not isinstance(self.loftq_config, dict):
            self.loftq_config = vars(self.loftq_config)