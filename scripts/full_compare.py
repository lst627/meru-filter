# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Perform image traversals using a trained MERU or CLIP model, and a pool of
text (and their encoded text representations).
"""
from __future__ import annotations

import argparse
import json
import os
import io

import torch
from PIL import Image
from torchvision import transforms as T

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager

from accelerate import Accelerator

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
# os.environ['HF_DATASETS_OFFLINE '] = "1" 
import datasets as ds
import matplotlib.pyplot as plt 
import webdataset as wds


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--dataset-name", help="mscoco, datacomp-small.")
_AA("--meru-checkpoint-path", default="/gscratch/sewoong/ericsf/meru-filter/meru_vit_l.pth", help="Path to checkpoint of a trained MERU model.")
_AA("--clip-checkpoint-path", default="/gscratch/sewoong/ericsf/meru-filter/clip_vit_l.pth", help="Path to checkpoint of a trained CLIP model.")
_AA("--dataset-path", default="/gscratch/scrubbed/ericsf/datacomp/shards/", help="Path to the image-text dataset.")
_AA("--fraction", default=1)
_AA("--reverse", default=False, action="store_true")
# /local1/datasets/datacomp_small or /local1/multi-modal-datasets/mscoco/
_AA("--meru-train-config", default="/gscratch/sewoong/ericsf/meru-filter/configs/train_meru_vit_l.py", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--clip-train-config", default="/gscratch/sewoong/ericsf/meru-filter/configs/train_clip_vit_l.py", help="Path to train config (.yaml/py) for given checkpoint.")

image_transform = T.Compose(
    [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
)

def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.
    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    if isinstance(model, MERU):
        # using meru scores
        # scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())
        # using time dimension
        scores = [[]]
        for x in text_feats:
            x_time = torch.sqrt(1 / model.curv.exp() + torch.sum(x**2, dim=-1, keepdim=True))
            scores[0].append(x_time)

        # For MERU, exclude text embeddings that do not entail the given image.
        # _aper = L.half_aperture(text_feats, model.curv.exp())
        # _oxy_angle = L.oxy_angle(
        #     text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        # )
        # entailment_energy = _oxy_angle - _aper[..., None]

        # # Root entails everything.
        # if has_root:
        #     entailment_energy[-1, ...] = 0

        # Set a large negative score if text does not entail image.  
        # scores[entailment_energy.T > 0] = -1e12
        return scores
    else:
        # model is not needed here.
        return image_feats @ text_feats.T

@torch.inference_mode()
def loader(dataset_name):
    if dataset_name == "mscoco":
        ds.config.DOWNLOADED_DATASETS_PATH = Path(_A.dataset_path)
        ds.config.HF_DATASETS_CACHE = Path(_A.dataset_path)
        dataset = ds.load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT")
        return dataset['train'], None
    elif dataset_name == "datacomp-small":
        dataset = wds.WebDataset(os.path.join(_A.dataset_path,"shards/00000564.tar")) \
                     .decode("pil") \
                     .to_tuple("json", "jpg;png", "txt") \
                     .map_tuple(lambda x: x["uid"], image_transform, lambda x: x)
        return dataset, None
    else:
        return None, None

@torch.inference_mode()
def main(_A: argparse.Namespace):
    # Get the current device (this will be `cuda:0` here by default) or use CPU.
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN_MERU = LazyConfig.load(_A.meru_train_config)
    model_meru = LazyFactory.build_model(_C_TRAIN_MERU, device).eval()

    CheckpointManager(model=model_meru).load(_A.meru_checkpoint_path)

    # root_feat_meru = torch.zeros(_C_TRAIN_MERU.model.embed_dim, device=device)
    # root_feat_clip = torch.load(_A.clip_checkpoint_path)["root"].to(device)

    # get all files ending in .tar in the dataset path
    tar_files = [f for f in os.listdir(_A.dataset_path) if f.endswith('.tar')]
    # sort the tar files by their number
    tar_files.sort(key=lambda x: int(x.split(".")[0]))
    # divide the tar files into 4 lists of equal length
    tar_files = [tar_files[i::4] for i in range(4)]
    # select the group of tar files corresdponding to the fraction
    tar_files = tar_files[int(_A.fraction)]
    print(f"Using split {_A.fraction} of the data, with {len(tar_files)} tar files and first tar file {tar_files[0]}")

    if _A.reverse:
        print("Reversing the order of tar files")
        tar_files = list(reversed(tar_files))

    tokenizer = Tokenizer()

    for tar_file in tqdm(tar_files, total=len(tar_files)):
        print(f"Processing tar file {tar_file}")
        # MSCOCO
        # print(len(train_data)) 118287
        # print(len(test_data)) 5000
        dataset = wds.WebDataset(os.path.join(_A.dataset_path, tar_file)) \
            .decode("pil") \
            .to_tuple("json", "jpg;png", "txt") \
            .map_tuple(lambda x: x["uid"], image_transform, lambda x: x)
        dataloader = wds.WebLoader(dataset, batch_size=256, num_workers=4)

        meru_score_collection = []
        meru_uid_collection = []

        for batch in dataloader:
            uids = batch[0]
            meru_uid_collection.extend(uids)
            img_tr = batch[1].to(device)
            txts = batch[2]
            img_feat = model_meru.encode_image(img_tr, project=True)[0]
            txt_tokenized = tokenizer(txts)
            txt_feats = model_meru.encode_text(txt_tokenized, project=True)
            scores_meru = calc_scores(model_meru, img_feat, txt_feats, has_root=False)

            # img_feat = model_clip.encode_image(img_tr, project=True)[0]
            # txt_feats = model_clip.encode_text(txt_tokenized, project=True)
            # scores_clip = calc_scores(model_clip, img_feat, txt_feats, has_root=False)

            meru_score_stack = torch.stack(scores_meru[0], dim=1).squeeze(0)
            for score_meru in meru_score_stack:
                meru_score_collection.append(score_meru.item())
            # for score_meru, score_clip in zip(meru_score_stack, scores_clip):
            #     meru_score_collection.append(score_meru.item())
            #     clip_score_collection.append(score_clip.item())

        assert len(meru_score_collection) == len(meru_uid_collection)
        print(f"Sample meru score: {meru_score_collection[0]}")
        print(f"Sample uid: {meru_uid_collection[0]}")
        print(f"Saves meru scores and uids to /gscratch/scrubbed/ericsf/datacomp/scores/meru-uids-and-xtimes-{tar_file}.pt")
        torch.save(zip(meru_uid_collection, meru_score_collection), f"/gscratch/scrubbed/ericsf/datacomp/scores/meru-uids-and-xtimes-{tar_file}.pt")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)