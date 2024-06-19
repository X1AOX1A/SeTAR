import math
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from svd_ood.models.swinv2 import Swinv2ForImageClassification

def train_preprocess(n_px, interpolation=InterpolationMode.BICUBIC):
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

def val_preprocess(n_px, interpolation=InterpolationMode.BICUBIC, crop_pct=0.9):
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

def load_model(model_name, device):
    model = Swinv2ForImageClassification.from_pretrained(model_name).to(device)
    train_processor = train_preprocess(model.config.image_size)
    val_processor = val_preprocess(model.config.image_size)
    tokenizer = None  # Swin Transformer does not have a text tower, so no tokenizer
    return model, train_processor, val_processor, tokenizer


if __name__ == "__main__":
    from PIL import Image
    import requests

    model_name = "microsoft/swinv2-base-patch4-window16-256"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_processor, val_processor, _ = load_model(model_name, device)
    model.enable_local_feat()
    model.eval()

    image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = val_processor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        logits, logits_local = outputs.logits, outputs.logits_local
        probs = logits.softmax(dim=1)
        top3_probabilities, top3_class_indices = torch.topk(probs * 100, k=3)
    print("Global logits shape:", list(logits.size()))
    print("Local logits shape:", list(logits_local.size()))
    print("Top 3 probs:", top3_probabilities.cpu().numpy())
    # Top 3 probs: [[79.87883    0.6606984  0.6579482]]
    print("Top 3 class indices:", top3_class_indices.cpu().numpy())
    # Top 3 class indices: [[918 409 688]]