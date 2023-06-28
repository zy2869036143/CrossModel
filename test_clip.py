import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("images/hashiqi.png")).unsqueeze(0).to(device)
text = clip.tokenize(["小可爱", "狗", "猫"]).to(device)
av = clip.tokenize(["这是一只可爱的小狗，它是一只哈士奇"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    # x, image_features = model.encode_text(av)
    text_ori, text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]