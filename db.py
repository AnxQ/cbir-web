import faiss
import faiss.contrib.torch_utils
import mmcv
import numpy as np
from typing import List, Union
import torch
import os


class FaissDatabase:
    def __init__(self,
                 dimensions=768,
                 index_method="IndexIVFFlat",
                 index_path=None,
                 meta_path=None):
        if index_path is not None:
            self.index = faiss.read_index(index_path)
        else:
            self.quantizer = faiss.IndexFlatL2(dimensions)
            self.index = getattr(faiss, index_method)(self.quantizer, dimensions, 100, faiss.METRIC_L2)

        if meta_path is not None:
            self.meta_dict = mmcv.load(meta_path)
            assert isinstance(self.meta_dict, dict)
        else:
            self.meta_dict = {}

        self.current_id = len(self.meta_dict)

    def save(self,
             index_path="checkpoints/db_index.idx",
             meta_path="checkpoints/db_meta.pkl"):
        faiss.write_index(self.index, index_path)
        mmcv.dump(self.meta_dict, meta_path)

    def train(self, features: Union[np.ndarray, torch.Tensor]):
        self.index.train(features)

    def add(self,
            features: Union[np.ndarray, torch.Tensor],
            names: list):
        ids = np.arange(self.current_id, self.current_id + len(names))
        self.meta_dict.update(dict(zip(ids, names)))
        self.current_id = self.current_id + len(names)
        self.index.add_with_ids(features, ids)

    def retrieve(self, feature, top_k=10):
        scores, neighbors = self.index.search(feature, top_k)
        return scores.flatten(), [*map(lambda idx: self.meta_dict[idx], neighbors.flatten())]


if __name__ == "__main__":
    prefix = "data/mini_imagenet/"
    db = FaissDatabase()
    features = np.load("data/features.npy")
    names = mmcv.load("data/names.pkl")
    names = [*map(lambda fn: os.path.basename(fn), names)]
    db.train(features)
    db.add(features, names)
    db.save()