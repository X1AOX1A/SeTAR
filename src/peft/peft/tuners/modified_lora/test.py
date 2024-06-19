import torch
from transformers import CLIPModel, CLIPTokenizer
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from peft import PeftModel, ModifiedLoraConfig

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

def default_clip_preprocess(n_px=224, interpolation="bilinear"):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
        transforms.Resize(n_px, interpolation=INTERPOLATION_MODES[interpolation]),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess

def load_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device)
    preprocess = default_clip_preprocess(model.config.vision_config.image_size)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, preprocess, tokenizer


if __name__ == "__main__":
    from svd_ood.utils.logger import setup_logger
    setup_logger(log_file=None)

    from PIL import Image
    import requests

    model_name = "openai/clip-vit-base-patch16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_model(model_name, device)
    model.eval()

    target_modules = {
        "vision_model.encoder.layers.6.self_attn.q_proj": 60,
        "vision_model.encoder.layers.5.self_attn.k_proj": 50,
        "vision_model.encoder.layers.4.self_attn.v_proj": 40,
        "vision_model.encoder.layers.3.self_attn.out_proj": 30,
        "vision_model.encoder.layers.2.mlp.fc1": 20,
        "vision_model.encoder.layers.1.mlp.fc2": 10,
        "visual_projection": 0,
        "text_model.encoder.layers.6.self_attn.q_proj": 0,
        "text_model.encoder.layers.5.self_attn.k_proj": 10,
        "text_model.encoder.layers.4.self_attn.v_proj": 20,
        "text_model.encoder.layers.3.self_attn.out_proj": 30,
        "text_model.encoder.layers.2.mlp.fc1": 40,
        "text_model.encoder.layers.1.mlp.fc2": 50,
        "text_projection": 60,
    }

    image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    text = torch.tensor(tokenizer(["a diagram", "a dog", "a cat"]).input_ids).to(device)

    def get_probs():
        with torch.no_grad():
            outputs = model(text, image)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            print(probs)

    # Original model
    get_probs() # [[0.76336074 0.22587983 0.01075946]]

    # Add Lora layers, lora_A (kaiming_normal) and lora_B (zeros)
    config = ModifiedLoraConfig(target_modules=target_modules)
    model = PeftModel(model, config)
    get_probs() # [[0.76336074 0.22587983 0.01075946]], lora_B = 0, so no change
    model.print_trainable_parameters()
    # trainable params: 744,960 || all params: 150,365,697 || trainable%: 0.49543214633587607
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, list(p.size()))
    # base_model.model.text_model.encoder.layers.1.mlp.fc2.lora_A.default.weight [50, 2048]
    # base_model.model.text_model.encoder.layers.1.mlp.fc2.lora_B.default.weight [512, 50]
    # base_model.model.text_model.encoder.layers.2.mlp.fc1.lora_A.default.weight [40, 512]
    # base_model.model.text_model.encoder.layers.2.mlp.fc1.lora_B.default.weight [2048, 40]
    # base_model.model.text_model.encoder.layers.3.self_attn.out_proj.lora_A.default.weight [30, 512]
    # base_model.model.text_model.encoder.layers.3.self_attn.out_proj.lora_B.default.weight [512, 30]
    # base_model.model.text_model.encoder.layers.4.self_attn.v_proj.lora_A.default.weight [20, 512]
    # base_model.model.text_model.encoder.layers.4.self_attn.v_proj.lora_B.default.weight [512, 20]
    # base_model.model.text_model.encoder.layers.5.self_attn.k_proj.lora_A.default.weight [10, 512]
    # base_model.model.text_model.encoder.layers.5.self_attn.k_proj.lora_B.default.weight [512, 10]
    # base_model.model.vision_model.encoder.layers.1.mlp.fc2.lora_A.default.weight [10, 3072]
    # base_model.model.vision_model.encoder.layers.1.mlp.fc2.lora_B.default.weight [768, 10]
    # base_model.model.vision_model.encoder.layers.2.mlp.fc1.lora_A.default.weight [20, 768]
    # base_model.model.vision_model.encoder.layers.2.mlp.fc1.lora_B.default.weight [3072, 20]
    # base_model.model.vision_model.encoder.layers.3.self_attn.out_proj.lora_A.default.weight [30, 768]
    # base_model.model.vision_model.encoder.layers.3.self_attn.out_proj.lora_B.default.weight [768, 30]
    # base_model.model.vision_model.encoder.layers.4.self_attn.v_proj.lora_A.default.weight [40, 768]
    # base_model.model.vision_model.encoder.layers.4.self_attn.v_proj.lora_B.default.weight [768, 40]
    # base_model.model.vision_model.encoder.layers.5.self_attn.k_proj.lora_A.default.weight [50, 768]
    # base_model.model.vision_model.encoder.layers.5.self_attn.k_proj.lora_B.default.weight [768, 50]
    # base_model.model.vision_model.encoder.layers.6.self_attn.q_proj.lora_A.default.weight [60, 768]
    # base_model.model.vision_model.encoder.layers.6.self_attn.q_proj.lora_B.default.weight [768, 60]
    # base_model.model.text_projection.lora_A.default.weight [60, 512]
    # base_model.model.text_projection.lora_B.default.weight [512, 60]

    # Initialize the base weight and Lora weights using SVD
    model.svd_init(lora_weights="small")    # "small", "large", "random"
    get_probs() # [[0.60441524 0.3795139  0.01607091]], != before, since there is a scaling on lora
    model.print_trainable_parameters()
    # trainable params: 744,960 || all params: 150,365,697 || trainable%: 0.49543214633587607

    # Disable Lora scaling
    model.disable_lora_scaling()
    get_probs() # [[0.7633678  0.22587457 0.01075761]], this time is close to the first time,
                # but not exactly the same, since there is a svd reconstruction

    # Disable the lora layers (small weights), **used for search**
    model.disable_adapter_layers()
    get_probs() # [[0.5612728  0.42042524 0.01830197]]

    # Enable the lora layers (small weights)
    model.enable_adapter_layers()
    get_probs()  # [[0.7633678  0.22587457 0.01075761]]

    # Merge the lora layers to the base model, but keep the lora layers
    # **Used for inference**, which is quicker than unmerged model
    model.merge_adapter(())
    get_probs()  # [[0.7633678  0.22587457 0.01075761]]

    # Unmerge the lora layers
    model.unmerge_adapter()
    get_probs()  # [[0.7633678  0.22587457 0.01075761]]

    # Merge the lora layers to the base model, and delete the lora layers
    model.merge_and_unload()
    get_probs()  # [[0.7633668  0.22587559 0.01075765]]