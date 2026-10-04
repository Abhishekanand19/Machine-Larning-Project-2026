"""Checks geometry, leakage boundaries, and saved inference against real artifacts."""
import sys
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from utils import ROOT, FEATURES, TARGETS, DEFAULT_CONFIG, project
from generate_dataset import generate
from preprocessing import load_splits
from predict import PosePredictor

class PipelineTests(unittest.TestCase):
    def test_projection_convention(self):
        centered,_ = project([50,0,0,0])
        right,_ = project([50,5,0,0])
        up,_ = project([50,0,5,0])
        farther,_ = project([100,0,0,0])
        rolled,_ = project([50,0,0,30])
        self.assertTrue((right[:,0]>centered[:,0]).all())
        self.assertTrue((up[:,1]<centered[:,1]).all())
        self.assertLess(np.ptp(farther[:,0]),np.ptp(centered[:,0]))
        self.assertFalse(np.allclose(rolled,centered))
        self.assertFalse(project([-10,0,0,0])[1].any())
        self.assertFalse(project([20,100,0,0])[1].any())

    def test_generator_reproducible(self):
        config = dict(DEFAULT_CONFIG,samples=100)
        pd.testing.assert_frame_equal(generate(dict(config)),generate(dict(config)))

    def test_splits_and_scaler(self):
        frame,indices,scaler = load_splits()
        all_indices = np.concatenate(list(indices.values()))
        np.testing.assert_array_equal(np.sort(all_indices),np.arange(len(frame)))
        self.assertEqual(len(np.unique(all_indices)),len(frame))
        self.assertEqual([len(indices[k]) for k in ('train','validation','test')],[14000,3000,3000])
        np.testing.assert_allclose(scaler.mean_,frame.iloc[indices['train']][TARGETS].mean().to_numpy())

    def test_saved_predictions(self):
        frame,indices,_ = load_splits()
        inputs = frame.iloc[indices['test'][:5]][FEATURES].to_numpy()
        predictor = PosePredictor()
        batch = predictor.predict(inputs)
        self.assertEqual(batch.shape,(5,4))
        self.assertTrue(np.isfinite(batch).all())
        np.testing.assert_allclose(predictor.predict(inputs[0]),batch[0],rtol=1e-5,atol=1e-5)
        self.assertGreater(np.std(batch[:,0]),1)
        with self.assertRaises(ValueError):
            predictor.predict(np.zeros(41))
        with self.assertRaises(ValueError):
            predictor.predict(np.zeros(42))

    def test_learning(self):
        history = pd.read_csv(ROOT/'results/training_history.csv')
        summary = json.loads((ROOT/'results/evaluation_summary.json').read_text())
        self.assertLess(history.training_loss.min(),history.training_loss.iloc[0]*.1)
        self.assertLess(history.validation_loss.min(),history.validation_loss.iloc[0]*.1)
        self.assertTrue(all(value>.9 for value in summary['correlation'].values()))

if __name__ == '__main__':
    unittest.main()
