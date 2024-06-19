import torch
from svd_ood.models.clip import CLIPModel
from transformers import CLIPTokenizer

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

def load_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device)
    preprocess = default_clip_preprocess(model.config.vision_config.image_size)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return model, preprocess, tokenizer


if __name__ == "__main__":
    from PIL import Image
    import requests

    model_name = "openai/clip-vit-base-patch16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_model(model_name, device)
    model.eval()

    image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    text = torch.tensor(tokenizer(["a diagram", "a dog", "a cat"]).input_ids).to(device)

    with torch.no_grad():
        # Normal clip, no local features return by default
        global_image_features = model.get_image_features(image)
        print(global_image_features.shape)
        # torch.Size([1, 512])
        print(global_image_features[0][:10])
        # tensor([-0.0030, -1.2102,  0.6114, -0.3273,  0.2932, -0.2274, -0.0147,  0.0699,
        #          0.2282,  0.1366], device='cuda:0')

        # Enable local features, which will increase the computation time
        model.enable_local_feat()   # set model.compute_local_feat to True
        global_image_features, local_image_features = model.get_image_features(image)
        print(global_image_features.shape, local_image_features.shape)
        # torch.Size([1, 512]) torch.Size([1, 14, 14, 512])

        print(global_image_features[0][:10])
        # tensor([-0.0030, -1.2102,  0.6114, -0.3273,  0.2932, -0.2274, -0.0147,  0.0699,
        #          0.2282,  0.1366], device='cuda:0')
        print(local_image_features[0][0][0][:10])
        # tensor([-0.6969,  0.0683, -0.0511,  0.0473,  0.1744, -0.1630,  0.5785, -0.6499,
        #         -0.4978,  0.2288], device='cuda:0')
        model.disable_local_feat()  # set model.compute_local_feat to False

        model.enable_local_feat()
        outputs = model(text, image)
        # (images, texts), image-text cosine similarity (with logit_scale, not softmax)
        global_image_logits = outputs.logits_per_image
        print(global_image_logits.shape)    # torch.Size([1, 3])
        # (images, weight, height, texts), local image-text cosine similarity (with logit_scale, not softmax)
        local_image_logits = outputs.logits_per_image_local
        print(local_image_logits.shape)     # torch.Size([1, 14, 14, 3])

        probs = global_image_logits.softmax(dim=-1).cpu().numpy()
        print(probs)                        # [[0.76336074 0.22587983 0.01075946]]