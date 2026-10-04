from pathlib import Path
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f'p{i}_{part}' for i in range(1, 15) for part in ('x', 'y', 'visible')]
TARGETS = ['target_x', 'target_y', 'target_z', 'target_roll']
NAMES = ['nose', 'front_fuselage', 'center', 'rear_fuselage', 'tail',
         'left_wing_root', 'left_wing_mid', 'left_wing_tip',
         'right_wing_root', 'right_wing_mid', 'right_wing_tip',
         'left_stabilizer', 'right_stabilizer', 'vertical_tail']
# Body coordinates in metres: forward, right, up. Symmetric in lateral axis.
POINTS = np.array([[5,0,0], [3,0,0], [0,0,0], [-3,0,0], [-5,0,0],
                   [0,-1,0], [-1,-4,0], [-2,-7,0], [0,1,0], [-1,4,0],
                   [-2,7,0], [-4,-3,0], [-4,3,0], [-4,0,2]], dtype=float)
EDGES = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(6,7),(2,8),(8,9),
         (9,10),(3,11),(11,4),(3,12),(12,4),(3,13),(13,4)]
DEFAULT_CONFIG = dict(seed=42, samples=20000, width=1280, height=720,
                      fx=400.0, fy=400.0, cx=640.0, cy=360.0,
                      noise_std_px=2.0, occlusion_probability=0.1,
                      min_visible=8, ranges=[[20,100],[-20,20],[-10,10],[-45,45]])

def directories():
    for folder in ('data/raw','data/processed','models','results/dataset_samples','docs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

def seed_all(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def transform_points(pose):
    """Rotate the fixed geometry and translate into camera coordinates."""
    angle = np.deg2rad(pose[3])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return POINTS @ rotation.T + np.asarray(pose[:3])

def project(pose, config=DEFAULT_CONFIG):
    """Rotate about forward X, then translate; u=cx+fx*Y/X, v=cy-fy*Z/X."""
    camera = transform_points(pose)
    depth = camera[:,0]
    safe_depth = np.where(depth > 0, depth, 1)
    pixels = np.column_stack([config['cx'] + config['fx']*camera[:,1]/safe_depth,
                              config['cy'] - config['fy']*camera[:,2]/safe_depth])
    valid = ((depth > 0) & (pixels[:,0] >= 0) & (pixels[:,0] < config['width'])
             & (pixels[:,1] >= 0) & (pixels[:,1] < config['height']))
    return pixels, valid

def observe_pose(pose, config, rng):
    """Shared dataset/demo process. Returns ONLY 42 camera features.

    Draw order matches the original generator to preserve seed-42 observations.
    Callers reject results with fewer than min_visible keypoints.
    """
    pixels, visible = project(pose, config)
    pixels += rng.normal(0,config['noise_std_px'],pixels.shape)
    visible &= ((pixels[:,0]>=0)&(pixels[:,0]<config['width'])&
                (pixels[:,1]>=0)&(pixels[:,1]<config['height']))
    visible &= rng.random(14) >= config['occlusion_probability']
    triples = np.zeros((14,3))
    triples[visible,:2] = 2*pixels[visible]/[config['width'],config['height']]-1
    triples[visible,2] = 1
    return triples.ravel()

def simulate_scenario(pose, config, seed):
    """Observe an in-range custom pose; retry corruption, never change the pose."""
    pose = np.asarray(pose,dtype=float)
    limits = np.asarray(config['ranges'])
    if pose.shape != (4,) or not np.isfinite(pose).all():
        raise ValueError('Expected four finite pose values.')
    if ((pose<limits[:,0]) | (pose>limits[:,1])).any():
        raise ValueError('Custom pose is outside the supported dataset ranges.')
    rng = np.random.default_rng(seed)
    for _ in range(100):
        features = observe_pose(pose,config,rng)
        if features.reshape(14,3)[:,2].sum() >= config['min_visible']:
            return features
    raise ValueError('Not enough visible keypoints. Choose another scenario.')

def observation_figure(features, pose, config, title='Synthetic camera observation'):
    triples = np.asarray(features).reshape(14,3)
    visible = triples[:,2].astype(bool)
    pixels = (triples[:,:2]+1)*np.array([config['width'],config['height']])/2
    ideal, valid = project(pose, config)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    missing = ~visible & valid
    for ax in axes:
        for a,b in EDGES:
            if visible[a] and visible[b]:
                ax.plot(pixels[[a,b],0], pixels[[a,b],1], color='#7a91a3', linewidth=1)
        ax.scatter(pixels[visible,0],pixels[visible,1],color='#156082',label='Observed features',s=30)
        ax.scatter(ideal[missing,0],ideal[missing,1],color='#ae7160',marker='x',label='Missing (truth overlay)')
        ax.set(xlabel='u (pixels)',ylabel='v (pixels)')
        ax.set_aspect('equal')
    axes[0].set(xlim=(0,config['width']),ylim=(config['height'],0),title='Full camera frame')
    shown = np.vstack([pixels[visible],ideal[missing]])
    low, high = shown.min(axis=0), shown.max(axis=0)
    pad = max(12,float((high-low).max())*.25)
    axes[1].set(xlim=(low[0]-pad,high[0]+pad),ylim=(high[1]+pad,low[1]-pad),title='Zoomed keypoints')
    axes[1].legend(loc='best',fontsize=8)
    fig.suptitle(f'{title}\nX={pose[0]:.1f} m, Y={pose[1]:.1f} m, Z={pose[2]:.1f} m, Roll={pose[3]:.1f}°')
    fig.tight_layout()
    return fig
