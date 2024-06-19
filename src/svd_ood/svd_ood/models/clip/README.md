# CLIP

Source: [transformers-V4.37.2](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/clip/modeling_clip.py)

## Main changes:

- Add local features to CLIPModel, refer to [this paper](https://arxiv.org/pdf/2304.04521.pdf") for details.

    ```python
    # enable local features, which will increase the computation time
    model.enable_local_feat()
    image_features, local_features = model.get_image_features(**inputs)

    # disable local features, back to normal CLIP model (by default)
    model.disable_local_feat()
    image_features = model.get_image_features(**inputs) # default
    ```

## Usage

```python
import torch
import requests
from PIL import Image
# from transformers import CLIPModel
from svd_ood.models.clip import CLIPModel
from transformers import CLIPTokenizer, CLIPProcessor

# load model
model_name = "openai/clip-vit-base-patch16"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image_url = "https://github.com/openai/CLIP/raw/main/CLIP.png"
image = Image.open(requests.get(image_url, stream=True).raw)
texts = ["a diagram", "a dog", "a cat"]

## get image-text similarity score and label probabilities
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
# [[0.8198, 0.1690, 0.0112]]

## get image features
inputs = processor(images=image, return_tensors="pt").to(device)
image_features = model.get_image_features(**inputs)
print(image_features.shape)
# [1, 512]
print(image_features[0][:10])
# [-0.0084, -1.3005,  0.5458, -0.3049,  0.1404, -0.2752, -0.0215,  0.1004,
#   0.2482,  0.1560]

## **get (global) image features and local features**
# enable local features, which will increase the computation time
model.enable_local_feat()   # trun on when you need local features
inputs = processor(images=image, return_tensors="pt").to(device)
image_features, local_features = model.get_image_features(**inputs)
# disable local features, back to normal CLIP model
model.disable_local_feat()  # turn off when you don't need local features
print(image_features.shape)
# [1, 512]
print(image_features[0][:10])
# [-0.0084, -1.3005,  0.5458, -0.3049,  0.1404, -0.2752, -0.0215,  0.1004,
#   0.2482,  0.1560]
print(local_features.shape)
# [1, 14, 14, 512]
print(local_features[0][0][0][:10])
# [-0.6977,  0.0866, -0.0720,  0.0347,  0.1289, -0.1820,  0.5403, -0.6942,
#  -0.4208,  0.2753]

## get text features
inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
text_features = model.get_text_features(**inputs)
print(text_features.shape)
# [3, 512]
print(text_features[0][:10])
# [ 0.0708, -0.1329,  0.2588, -0.2482, -0.3146,  0.3575, -0.0310, -0.2971,
#   0.2386, -0.2773]
```