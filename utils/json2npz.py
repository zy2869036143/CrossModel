file_path = "../data/all_data.json"
import json
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import argparse
from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])

args.seed = 42
args.num_classes_layer = [28, 356]
args.total_classes = 384





device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../')
model.eval()

data = []
with open(file_path) as f:
    for line in f:
        js = json.loads(line)
        data.append(js)

def _create_onehot_labels(labels_index, num_labels):

    label = [0] * num_labels
    for item in labels_index:
        label[int(item)] = 1
    return label

for js in data:
    file_name = "../nps/" + str(js["id"]) + ".npz"
    if os.path.exists(file_name):
        print("Loading %d.npz file." % js["id"])
        continue

    if js['isImage'] == False:
        text_data = js['title'] + js['abstract']
        clip_token = clip.tokenize(text_data).to(device)  # Shape: (1,52)
        features, global_feature = model.encode_text(clip_token)
    # else:
        # TODO: Using CN-Clip to encode the image data.
        # tmp = js['abstract'][0].split('.')
        # if tmp[-1] in ["jpg", "jpeg", "png"]:
        #     path = js["abstract"][0]
        #     image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        #     global_feature = model.encode_image(image)
        #     global_feature = global_feature.repeat(53, 1)


        global_feature /= global_feature.norm(dim=-1, keepdim=True)

        # TODO: Change to tow level tags
        onehot_labels_tuple_list = (_create_onehot_labels(js['section'], args.num_classes_layer[0]),
                                    _create_onehot_labels(js['subsection'], args.num_classes_layer[1]))
        onehot_labels_list = (_create_onehot_labels(js['labels'], args.total_classes))

        feature_np = global_feature.squeeze().detach().cpu().numpy()
        del global_feature, features
        info = "Creating " + str(js["id"]) + ".npz file with feature shape: " + str(feature_np.shape)
        print(info)
        np.savez(file_name, id=js['id'], feature=feature_np,
                 labels=js['labels'], layer1=onehot_labels_tuple_list[0],
                 layer2=onehot_labels_tuple_list[1],
                 full=onehot_labels_list)