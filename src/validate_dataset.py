import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import ROOT, FEATURES, TARGETS, observation_figure, save_json

def validate(frame, config):
    assert frame.shape == (config['samples'],46), 'Wrong dataset dimensions'
    assert list(frame.columns) == FEATURES+TARGETS, 'Wrong schema/order'
    assert not frame.isna().any().any(), 'Null values'
    assert np.isfinite(frame.to_numpy()).all(), 'Non-finite values'
    assert frame.duplicated().sum() == 0, 'Duplicate rows'
    features = frame[FEATURES].to_numpy().reshape(-1,14,3)
    assert np.isin(features[:,:,2],[0,1]).all(), 'Invalid visibility flags'
    assert (np.abs(features[:,:,:2]) <= 1).all(), 'Coordinates outside normalized range'
    assert (features[:,:,:2][features[:,:,2]==0] == 0).all(), 'Missing coordinates must be zero'
    counts = features[:,:,2].sum(axis=1)
    assert counts.min() >= config['min_visible'], 'Insufficient visible keypoints'
    for column, (lo,hi) in zip(TARGETS,config['ranges']):
        assert frame[column].between(lo,hi).all(), f'Target range: {column}'
    report = dict(rows=len(frame),columns=len(frame.columns),nulls=0,nonfinite=0,duplicates=0,
                  minimum_visible=int(counts.min()),maximum_visible=int(counts.max()),
                  mean_visible=float(counts.mean()),missing_fraction=float(1-counts.mean()/14),
                  target_ranges={c:[float(frame[c].min()),float(frame[c].max())] for c in TARGETS},
                  visible_counts={str(int(k)):int(v) for k,v in zip(*np.unique(counts,return_counts=True))},
                  rejected_samples=config['rejected_samples'])
    save_json(ROOT/'results/dataset_validation.json',report)
    for n, index in enumerate(np.random.default_rng(42).choice(len(frame),6,replace=False)):
        row = frame.iloc[index]
        fig = observation_figure(row[FEATURES],row[TARGETS].to_numpy(),config,f'Sample {index}')
        fig.savefig(ROOT/f'results/dataset_samples/sample_{n+1}.png',dpi=140)
        plt.close(fig)
    print(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    validate(pd.read_csv(ROOT/'data/raw/formation_flying_synthetic.csv'),
             json.loads((ROOT/'data/raw/generation_config.json').read_text()))
