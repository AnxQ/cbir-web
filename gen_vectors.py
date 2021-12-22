import os
import numpy as np
import mmcv
from db import FaissDatabase
from models.swin_transformer import feature_generator

image_root = "data/image_db"
index_path = "checkpoints/db_index.idx"
meta_path = "checkpoints/db_meta.pkl"

# Prevent memory leak with low memory environments
model_batch_size = 32
db_batch_size = 320000

full_paths = [os.path.join(image_root, image) for image in os.listdir(image_root)]
batch_feats = []
batch_names = []
db = FaissDatabase(dimensions=768, index_method="IndexIVFFlat")
db_trained = False
for feats, names in feature_generator(full_paths, model_batch_size):
    batch_feats.append(feats[0].cpu())
    batch_names.append(names)
    if len(batch_names) == db_batch_size:
        features = np.vstack(batch_feats).astype(np.float32)
        if not db_trained:
            db.train(features)
            db_trained = True
        db.add(features, batch_names)
        batch_feats = []
        batch_names = []

features = np.vstack(batch_feats).astype(np.float32)
db.add(features, batch_names)
db.save(index_path=index_path, meta_path=meta_path)