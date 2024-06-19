import torch
from transformers import CLIPTokenizer
from svd_ood.models.locoop import LoCoOpModel

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
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

def load_model(model_name, n_ctx, locoop_ckpt, device):
    model = LoCoOpModel.from_pretrained(model_name).to(device)
    model.enable_local_feat()
    model.register_prompt(n_ctx)
    model.load_prompter(locoop_ckpt)
    preprocess = default_clip_preprocess(model.config.vision_config.image_size)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, preprocess, tokenizer


if __name__ == "__main__":
    from PIL import Image
    import requests

    model_name = "openai/clip-vit-base-patch16"
    locoop_ckpt="/data/MODELS/LoCoOp/checkpoints/seed2/prompt_learner/model.pth.tar-50"
    n_ctx=16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_model(model_name, n_ctx, locoop_ckpt, device)
    model.eval()

    image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    text = tokenizer(["a diagram.", "a dog.", "a cat."], return_tensors="pt").to(device)

    with torch.no_grad():
        # get features from each tower after projection
        print("Getting features from each tower after projection...")
        global_image_features, local_image_features = model.get_image_features(image)
        text_features = model.get_text_features(**text)
        print("global_image_features:", list(global_image_features.shape))  # (images, output_dim)
        # global_image_features: [1, 512]
        print("local_image_features:", list(local_image_features.shape))    # (images, weight, height, texts)
        # local_image_features: [1, 14, 14, 512]
        print("text_features:", list(text_features.shape))                  # (texts, output_dim)
        # text_features: [3, 512]

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
        # image_logits_global: [1, 3]
        print("image_logits_local:", list(image_logits_local.shape))    # (images, weight, height, texts)
        # image_logits_local: [1, 14, 14, 3]
        print("text_logits:", list(text_logits.shape))                  # (texts, images)
        # text_logits: [3, 1]

        probs = image_logits_global.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)
        # Label probs: [[0.88056386 0.11070069 0.00873542]]