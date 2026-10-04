"""Custom-scenario integration tests with the existing saved ANN and dataset."""
import sys
import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from utils import ROOT, DEFAULT_CONFIG, simulate_scenario, FEATURES, TARGETS
from generate_dataset import generate
from predict import PosePredictor
from demo_visuals import camera_view, formation_view, side_view, roll_view
from preprocessing import load_splits
from evaluate import regression_scores

class DemoTests(unittest.TestCase):
    def test_refactor_preserves_dataset(self):
        config = json.loads((ROOT/'data/raw/generation_config.json').read_text())
        generated = generate(dict(config))
        original = pd.read_csv(ROOT/'data/raw/formation_flying_synthetic.csv')
        np.testing.assert_allclose(generated.to_numpy(),original.to_numpy(),rtol=0,atol=1e-12)

    def test_custom_features_and_predictions(self):
        predictor = PosePredictor()
        captured = []
        hook = predictor.model.register_forward_pre_hook(lambda model,args: captured.append(args[0].clone().numpy()))
        scenarios = [[25,0,0,0],[90,0,0,0],[60,-15,0,0],[60,15,0,0],
                     [60,0,-8,0],[60,0,8,0],[60,0,0,-40],[60,0,0,40]]
        predictions = []
        for pose in scenarios:
            features = simulate_scenario(pose,DEFAULT_CONFIG,42)
            self.assertEqual(features.shape,(42,))
            self.assertGreaterEqual(features.reshape(14,3)[:,2].sum(),8)
            predictions.append(predictor.predict(features))
            np.testing.assert_array_equal(captured[-1],features.astype(np.float32)[None,:])
            # Visible plot coordinates are exactly the inverse-normalized ANN coordinates.
            fig = camera_view(features,DEFAULT_CONFIG)
            triples = features.reshape(14,3)
            detected = (triples[:,:2]+1)*[1280,720]
            detected /= 2
            visible = triples[:,2].astype(bool)
            np.testing.assert_allclose(fig.data[1].x,detected[visible,0])
            np.testing.assert_allclose(fig.data[1].y,detected[visible,1])
            self.assertEqual(len(fig.data),2)  # No truth overlay by default.
            full = camera_view(features,DEFAULT_CONFIG,full_frame=True)
            np.testing.assert_array_equal(full.layout.xaxis.range,[0,1280])
            np.testing.assert_array_equal(full.layout.yaxis.range,[720,0])
            self.assertGreaterEqual(min(detected[visible,0]),min(fig.layout.xaxis.range))
            self.assertLessEqual(max(detected[visible,0]),max(fig.layout.xaxis.range))
        hook.remove()
        for low,high,axis in [(0,1,0),(2,3,1),(4,5,2),(6,7,3)]:
            self.assertGreater(predictions[high][axis],predictions[low][axis])

    def test_camera_behavior(self):
        clean = dict(DEFAULT_CONFIG,noise_std_px=0,occlusion_probability=0)
        def pixels(pose):
            return (simulate_scenario(pose,clean,42).reshape(14,3)[:,:2]+1)*[640,360]
        center = pixels([60,0,0,0])
        self.assertGreater(np.ptp(pixels([25,0,0,0])[:,0]),np.ptp(pixels([90,0,0,0])[:,0]))
        self.assertTrue((pixels([60,15,0,0])[:,0]>pixels([60,-15,0,0])[:,0]).all())
        self.assertTrue((pixels([60,0,8,0])[:,1]<pixels([60,0,-8,0])[:,1]).all())
        self.assertGreater(pixels([60,0,0,-40])[10,1],pixels([60,0,0,40])[10,1])
        self.assertFalse(np.allclose(center,pixels([60,0,0,35])))

    def test_diagrams_use_true_and_predicted_pose(self):
        truth = [60,-15,8,-40]
        predicted = [63,-13,7,-38]
        top = formation_view(truth,predicted)
        side = side_view(truth,predicted)
        # The nose of each translated icon supplies a known reference point.
        self.assertEqual(top.data[1].x[0],truth[1])
        self.assertEqual(top.data[1].y[0],truth[0]+5)
        self.assertEqual(top.data[2].x[0],predicted[1])
        self.assertEqual(side.data[1].x[0],truth[0]+5)
        self.assertEqual(side.data[2].y[0],predicted[2])
        self.assertEqual(top.data[2].line.dash,'dash')
        self.assertFalse(np.allclose(roll_view(truth,predicted).data[0].y,
                                     roll_view([60,-15,8,40],predicted).data[0].y))

    def test_r2_uses_test_predictions(self):
        frame,indices,_ = load_splits()
        actual = frame.iloc[indices['test']][TARGETS].to_numpy()
        predicted = PosePredictor().predict(frame.iloc[indices['test']][FEATURES].to_numpy())
        manual = 1-((actual-predicted)**2).sum(axis=0)/((actual-actual.mean(axis=0))**2).sum(axis=0)
        scores = regression_scores(actual,predicted)
        np.testing.assert_allclose(list(scores.values()),manual)
        stored = json.loads((ROOT/'results/evaluation_summary.json').read_text())['r2']
        np.testing.assert_allclose(list(scores.values()),list(stored.values()))

if __name__ == '__main__':
    unittest.main()
