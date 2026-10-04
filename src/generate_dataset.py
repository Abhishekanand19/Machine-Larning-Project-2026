import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import ROOT, FEATURES, TARGETS, POINTS, NAMES, EDGES, DEFAULT_CONFIG, directories, observe_pose, save_json
from validate_dataset import validate

def generate(config):
    rng = np.random.default_rng(config['seed'])
    rows, attempted = [], 0
    limits = np.asarray(config['ranges'])
    while len(rows) < config['samples']:
        attempted += 1
        if attempted > config['samples']*100:
            raise ValueError('Too many rejected samples; check camera and visibility settings.')
        pose = rng.uniform(limits[:,0],limits[:,1])
        features = observe_pose(pose,config,rng)
        if features.reshape(14,3)[:,2].sum() < config['min_visible']:
            continue
        rows.append(np.r_[features,pose])
    config['attempted_samples'] = attempted
    config['rejected_samples'] = attempted-len(rows)
    return pd.DataFrame(rows,columns=FEATURES+TARGETS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=20000)
    parser.add_argument('--noise',type=float,default=2)
    parser.add_argument('--occlusion',type=float,default=.1)
    args = parser.parse_args()
    if args.samples < 20 or args.noise < 0 or not 0 <= args.occlusion < 1:
        parser.error('Require samples >=20, noise >=0, and 0 <= occlusion <1.')
    directories()
    config = dict(DEFAULT_CONFIG, samples=args.samples, noise_std_px=args.noise, occlusion_probability=args.occlusion)
    frame = generate(config)
    frame.to_csv(ROOT/'data/raw/formation_flying_synthetic.csv',index=False)
    save_json(ROOT/'data/raw/generation_config.json',config)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111,projection='3d')
    for a,b in EDGES:
        ax.plot(*POINTS[[a,b]].T,color='#7a91a3')
    ax.scatter(*POINTS.T,color='#156082')
    for i, point in enumerate(POINTS):
        ax.text(*point,f'{i+1}: {NAMES[i]}',fontsize=7)
    ax.set(xlabel='Body X: forward (m)',ylabel='Body Y: right (m)',zlabel='Body Z: up (m)',title='Fixed 14-point aircraft geometry')
    ax.set_box_aspect((10,14,5))
    fig.tight_layout()
    fig.savefig(ROOT/'results/aircraft_keypoints.png',dpi=160)
    plt.close(fig)
    validate(frame,config)

if __name__ == '__main__':
    main()
