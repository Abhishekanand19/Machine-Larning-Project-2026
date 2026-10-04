import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from utils import ROOT, FEATURES, TARGETS, directories, save_json

def dataset_hash():
    return hashlib.sha256((ROOT/'data/raw/formation_flying_synthetic.csv').read_bytes()).hexdigest()

def prepare():
    directories()
    frame = pd.read_csv(ROOT/'data/raw/formation_flying_synthetic.csv')
    order = np.random.default_rng(42).permutation(len(frame))
    a,b = int(.7*len(frame)),int(.85*len(frame))
    indices = dict(train=order[:a],validation=order[a:b],test=order[b:])
    scaler = StandardScaler().fit(frame.iloc[indices['train']][TARGETS])
    joblib.dump(scaler,ROOT/'models/target_scaler.joblib')
    np.savez(ROOT/'data/processed/split_indices.npz',**indices)
    save_json(ROOT/'data/processed/split_metadata.json',dict(seed=42,dataset_sha256=dataset_hash(),
              sizes={k:len(v) for k,v in indices.items()},feature_columns=FEATURES,target_columns=TARGETS))
    print({k:len(v) for k,v in indices.items()})
    return frame,indices,scaler

def load_splits():
    import json
    metadata = json.loads((ROOT/'data/processed/split_metadata.json').read_text())
    if metadata['dataset_sha256'] != dataset_hash():
        raise ValueError('Dataset changed. Rerun preprocessing and training.')
    return (pd.read_csv(ROOT/'data/raw/formation_flying_synthetic.csv'),
            dict(np.load(ROOT/'data/processed/split_indices.npz')),
            joblib.load(ROOT/'models/target_scaler.joblib'))

if __name__ == '__main__':
    prepare()
