import sys
from pathlib import Path
if __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import joblib
from model import PoseANN
from utils import ROOT, FEATURES, TARGETS
from preprocessing import dataset_hash

class PosePredictor:
    def __init__(self):
        checkpoint = torch.load(ROOT/'models/best_model.pt',map_location='cpu',weights_only=True)
        if checkpoint['dataset_sha256'] != dataset_hash():
            raise ValueError('Model and dataset differ. Rerun training.')
        if checkpoint['feature_columns'] != FEATURES or checkpoint['target_columns'] != TARGETS:
            raise ValueError('Model schema differs.')
        self.model = PoseANN()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.scaler = joblib.load(ROOT/'models/target_scaler.joblib')

    def predict(self, observation):
        values = np.asarray(observation,dtype=np.float32)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.ndim != 2 or values.shape[1] != 42 or not np.isfinite(values).all():
            raise ValueError('Expected finite observations with exactly 42 features.')
        triples = values.reshape(-1,14,3)
        if not np.isin(triples[:,:,2],[0,1]).all() or (np.abs(triples[:,:,:2])>1).any():
            raise ValueError('Coordinates must be in [-1,1] and visibility binary.')
        if (triples[:,:,:2][triples[:,:,2]==0]!=0).any() or (triples[:,:,2].sum(axis=1)<8).any():
            raise ValueError('Zero missing coordinates and at least eight visible features are required.')
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(values)).numpy()
        poses = self.scaler.inverse_transform(outputs)
        return poses[0] if single else poses

if __name__ == '__main__':
    from preprocessing import load_splits
    frame,indices,_ = load_splits()
    predictor = PosePredictor()
    for index in indices['test'][:5]:
        row = frame.iloc[index]
        print(f'Test row {index}: truth={row[TARGETS].to_numpy().round(2)}, prediction={predictor.predict(row[FEATURES]).round(2)}')
