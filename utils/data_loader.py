import json
from PIL import Image
import torch
import torch.optim
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"


class InputFeature(object):
    def __init__(self,
                 _id,
                 features,
                 labels,
                 onehot_labels_tuple_list,
                 onehot_labels_list) -> None:
        self.id = _id
        self.features = features
        self.labels = labels
        self.onehot_labels_tuple_list = onehot_labels_tuple_list
        self.onehot_labels_list = onehot_labels_list


def convert_examples_to_features(js, args, clip, model, preprocess):
    # def _pad_features(texts_ints, seq_length):
    #     features = np.zeros((1, seq_length), dtype=int)
    #
    #     features[0, -texts_ints.shape[1]:] = np.array(texts_ints)[:seq_length]
    #
    #     # for i, row in enumerate(texts_ints):
    #     #     features[i, -len(row):] = np.array(row)[:seq_length]
    #     return features.reshape(-1)

    def _create_onehot_labels(labels_index, num_labels):

        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label


    if not os.path.exists("./nps"):
        os.makedirs("./nps")
    file_name = "./nps/" + str(js["id"]) + ".npz"

    if os.path.exists(file_name):
        print("Loading %d.npz file." % js["id"])
        # npz_file = np.load(file_name)
        return False
        # return InputFeature(npz_file["id"], npz_file["feature"], npz_file["labels"],
        #                     (npz_file["layer1"], npz_file["layer2"]), npz_file["full"])


    # TODO: Using CN-Clip to encode the text data.
    if js['isImage'] == False:
        text_data = js['title'] + js['abstract']
        clip_token = clip.tokenize(text_data).to(device)  # Shape: (1,52)
        features, global_feature = model.encode_text(clip_token)
    else:
        return False
        # TODO: Using CN-Clip to encode the image data.
        # tmp = js['abstract'][0].split('.')
        # if tmp[-1] in ["jpg", "jpeg", "png"]:
        #     path = js["abstract"][0]
        #     image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        #     global_feature = model.encode_image(image)
        #     global_feature = global_feature.repeat(53, 1)
        # else:
        #     return False

    global_feature /= global_feature.norm(dim=-1, keepdim=True)

    # TODO: Change to tow level tags
    onehot_labels_tuple_list = (_create_onehot_labels(js['section'], args.num_classes_layer[0]),
                                _create_onehot_labels(js['subsection'], args.num_classes_layer[1]))
    onehot_labels_list = (_create_onehot_labels(js['labels'], args.total_classes))


    feature_np = global_feature.squeeze().detach().cpu().numpy()
    info = "Creating "+str(js["id"])+ ".npz file with feature shape: " + str(feature_np.shape)
    print(info)
    logging.info(info)
    np.savez(file_name, id=js['id'], feature=feature_np,
            labels=js['labels'], layer1=onehot_labels_tuple_list[0],
             layer2=onehot_labels_tuple_list[1],
             full=onehot_labels_list)
    return False
    # return InputFeature(js['id'], global_feature.squeeze().detach().cpu().numpy(), js['labels'],
    #                     onehot_labels_tuple_list, onehot_labels_list)

class TextDataset(Dataset):

    def __init__(self, args, file_path, clip, model, preprocess) -> None:
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line)
                data.append(js)



        for js in data:
            fea = convert_examples_to_features(js, args, clip, model, preprocess)
            if not fea:
                continue
            self.examples.append(fea)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        # TODO: Change to tow level tags
        return (torch.tensor(self.examples[index].features),
                torch.tensor(self.examples[index].onehot_labels_list),
                torch.tensor(self.examples[index].onehot_labels_tuple_list[0]),
                torch.tensor(self.examples[index].onehot_labels_tuple_list[1]))

# args={'file_path':'data/validation_sample.json', 'seq_length':200, 'num_classes_layer':[9, 128, 661, 8364], 'total_classes':9162}
# dataset = TextDataset(args, args['file_path'])
# print(dataset.__len__())
