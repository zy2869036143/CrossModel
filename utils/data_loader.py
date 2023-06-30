import json
from collections import Counter

import torch
import torch.optim
from torch.utils.data.dataset import Dataset

import numpy as np
from utils import data_helper
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

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

def convert_examples_to_features(js, args, vocab_to_int):
    
    def _pad_features(texts_ints, seq_length):
        features = np.zeros((1, seq_length), dtype=int)
        
        features[0,-texts_ints.shape[1]:] = np.array(texts_ints)[:seq_length]

        # for i, row in enumerate(texts_ints):
        #     features[i, -len(row):] = np.array(row)[:seq_length]
        return features.reshape(-1)


    def _create_onehot_labels(labels_index, num_labels):
        
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label


    # TODO: Using CN-Clip to encode the text data.
    text_data = ' '.join(js['title'] + js['abstract'])

    # Original Codes
    # texts_ints = np.array([vocab_to_int[word] for word in text_data.split()]).reshape(1, -1)
    # features = _pad_features(texts_ints,seq_length=args.seq_length)

    # Changes been made.
    clip_token = clip.tokenize(text_data).to(args.device) # Shape: (1,52)
    features, global_feature = model.encode_text(clip_token)

    # TODO: Change to tow level tags

    # onehot_labels_tuple_list = (_create_onehot_labels(js['section'], args.num_classes_layer[0]),
    #                             _create_onehot_labels(js['subsection'], args.num_classes_layer[1]),
    #                             _create_onehot_labels(js['group'], args.num_classes_layer[2]),
    #                             _create_onehot_labels(js['subgroup'], args.num_classes_layer[3]))
    onehot_labels_tuple_list = (_create_onehot_labels(js['section'], args.num_classes_layer[0]),
                                _create_onehot_labels(js['subsection'], args.num_classes_layer[1]))
    onehot_labels_list = (_create_onehot_labels(js['labels'], args.total_classes))
    return InputFeature(js['id'], features.data.cpu().squeeze().detach().numpy(), js['labels'], onehot_labels_tuple_list, onehot_labels_list )


class TextDataset(Dataset):
    def __init__(self, args, file_path) -> None:
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line)
                data.append(js)

        # TODO: Change the vocabulary dictionary using the original of CN-Clip.
        vocab_to_int = data_helper.create_vocab(data)
        self.vocab_size = len(vocab_to_int)

        for js in data:
            self.examples.append(convert_examples_to_features(js, args, vocab_to_int))

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