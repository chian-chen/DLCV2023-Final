import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model1, _ = clip.load("ViT-B/16", device=device)    #  ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

text = ["a photo of a cat","a photo of a dog","a photo of a mug"]
text = text[0]
print(text)
text = clip.tokenize(text).to(device)
model2, preprocess = clip.load("ViT-B/32", device=device)
image1 = preprocess(Image.open("./CLIP/cat.png")).unsqueeze(0).to(device)
image2 = preprocess(Image.open("./CLIP/dog.jpg")).unsqueeze(0).to(device)
image3 = preprocess(Image.open("./CLIP/mug.jpg")).unsqueeze(0).to(device)
images = torch.cat([image1, image2, image3], dim=0)
# print(images.shape)
with torch.no_grad():   
    text_features = model1.encode_text(text)
    image_features = model2.encode_image(images)
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

print(text_features.shape)
print(image_features.shape)
if text_features.shape[-1] == image_features.shape[-1]:
    print(text_features[0,:]@image_features[0,:])
    print(text_features[0,:]@image_features[1,:])
    print(text_features[0,:]@image_features[2,:])
    if text_features.shape[0] > image_features.shape[0]:
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    else:
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)

    print(similarity)
# print(probs)

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]