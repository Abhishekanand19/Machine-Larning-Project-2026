"""Interactive demo: physical scenario -> camera features -> saved ANN."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from utils import ROOT, FEATURES, TARGETS, simulate_scenario
from preprocessing import load_splits
from predict import PosePredictor
from evaluate import regression_scores
from demo_visuals import formation_view, side_view, roll_view, camera_view, loss_view
from demo_controls import PRESETS, POSE_KEYS, human_pose

st.set_page_config(page_title='Formation Flying', layout='wide')
st.title('Formation Flying')
st.caption('ML-Based Relative Pose Estimation · synthetic camera observations')
page = st.radio('Page', ['Home', 'Try the Model', 'Model Performance', 'How It Works'],
                index=1, horizontal=True, label_visibility='collapsed')
required = ['models/best_model.pt', 'models/target_scaler.joblib', 'data/processed/split_indices.npz',
            'results/training_summary.json', 'results/training_history.csv', 'data/raw/generation_config.json']
if any(not (ROOT/file).exists() for file in required):
    st.info('Run dataset generation and training using the README commands before opening the demo.')
    st.stop()

def stamp(file):
    return (ROOT/file).stat().st_mtime_ns

@st.cache_resource
def load_predictor(model_stamp, scaler_stamp, dataset_stamp):
    return PosePredictor()

@st.cache_data
def load_data(dataset_stamp, split_stamp):
    frame, indices, _ = load_splits()
    return frame, indices

@st.cache_data
def held_out_scores(model_stamp, scaler_stamp, dataset_stamp, split_stamp):
    frame, indices = load_data(dataset_stamp, split_stamp)
    test = frame.iloc[indices['test']]
    predictor = load_predictor(model_stamp, scaler_stamp, dataset_stamp)
    predictions = predictor.predict(test[FEATURES].to_numpy())
    return regression_scores(test[TARGETS].to_numpy(), predictions)

try:
    frame, indices = load_data(stamp('data/raw/formation_flying_synthetic.csv'), stamp('data/processed/split_indices.npz'))
    predictor = load_predictor(stamp('models/best_model.pt'), stamp('models/target_scaler.joblib'),
                               stamp('data/raw/formation_flying_synthetic.csv'))
except (ValueError, FileNotFoundError) as error:
    st.error(str(error))
    st.stop()
config = json.loads((ROOT/'data/raw/generation_config.json').read_text())
training = json.loads((ROOT/'results/training_summary.json').read_text())

if page == 'Home':
    st.subheader('The problem')
    st.write('A follower aircraft needs to estimate the relative pose of a leader aircraft.')
    st.code('Leader Aircraft → Follower Camera → 14 Keypoints → 42 Features → ANN → X, Y, Z, Roll', language=None)
    for column, title, text in zip(st.columns(3), ['Input', 'Machine learning', 'Output'],
                                  ['14 projected structural keypoints, each with x, y and visibility.',
                                   'A feed-forward neural network processes 42 camera features.',
                                   'Forward distance X, lateral Y, vertical Z, and roll.']):
        column.subheader(title)
        column.write(text)
    st.info('Open Try the Model to choose a physical scenario and test the ANN.')
    st.caption('This is a synthetic ML experiment. Real image detection and flight control are outside scope.')

elif page == 'Try the Model':
    st.subheader('1. Choose a Formation Scenario')
    mode = st.radio('Observation source', ['Custom Flight Scenario', 'Unseen Test Sample'], horizontal=True)
    if mode == 'Custom Flight Scenario':
        for key,value in zip(POSE_KEYS,[50.,0.,0.,0.]):
            st.session_state.setdefault(key,value)
        def apply_preset(name):
            for key,value in zip(POSE_KEYS,PRESETS[name]):
                st.session_state[key] = value
        st.caption('Quick scenarios — choose one, then generate the camera observation.')
        for column,name in zip(st.columns(5),PRESETS):
            column.button(name,on_click=apply_preset,args=(name,),width='stretch')
        with st.form('custom_scenario'):
            left, right = st.columns(2)
            x = left.slider('Distance ahead (X)',20.,100.,step=1.,key='scene_x',format='%g m ahead')
            y = right.slider('Left / Right offset (Y)',-20.,20.,step=1.,key='scene_y',format='%g m',
                             help='Negative = left. Positive = right.')
            z = left.slider('Height difference (Z)',-10.,10.,step=.5,key='scene_z',format='%g m',
                            help='Negative = below. Positive = above.')
            roll = right.slider('Aircraft bank / roll',-45.,45.,step=1.,key='scene_roll',format='%g°',
                                help='Negative = left bank. Positive = right bank.')
            left.caption('X: 20–100 m ahead · Z: negative below / positive above')
            right.caption('Y: negative left / positive right · roll: negative left bank / positive right bank')
            st.caption('Selected values (updated on preset selection or submission): ' + ' · '.join(human_pose([x,y,z,roll])))
            with st.expander('Advanced simulation settings'):
                noise = st.slider('Pixel noise (standard deviation)',0.,5.,float(config['noise_std_px']),.5,format='%g px')
                missing = st.slider('Missing-feature probability',0.,.25,float(config['occlusion_probability']),.01)
                seed = int(st.number_input('Observation seed',min_value=0,max_value=2**32-1,value=42,step=1))
                st.caption('Defaults match training. A fixed seed makes scenario comparisons repeatable.')
            st.info('You choose the true flight situation. These values are used only to simulate what the follower '
                    'camera sees. The ANN receives only the resulting camera keypoints — not X, Y, Z or Roll.')
            submitted = st.form_submit_button('Generate Camera View & Predict', type='primary')
        if submitted:
            truth = np.array([x, y, z, roll])
            simulation_config = dict(config,noise_std_px=noise,occlusion_probability=missing)
            try:
                features = simulate_scenario(truth,simulation_config,seed)
                # Only camera features enter the ANN. Never concatenate the true pose.
                prediction = predictor.predict(features)
                st.session_state.custom_result = dict(truth=truth,features=features,prediction=prediction,seed=seed,
                                                     noise=noise,missing=missing)
            except ValueError as error:
                st.error(str(error))
        result = st.session_state.get('custom_result')
        if result is None:
            st.info('Choose a pose, then press Generate Camera View & Predict.')
            st.stop()
        st.success('Generated scenario: ' + ' · '.join(human_pose(result['truth'])))
        st.caption('The diagrams and output below show this submitted scenario. Changing a preset or slider does not '
                   'change it until you press Generate. '
                   f"Noise {result['noise']:g} px · missingness {100*result['missing']:g}% · seed {result['seed']}.")
    else:
        st.caption('This mode uses only rows from the saved held-out test split.')
        if 'test_sample_id' not in st.session_state:
            st.session_state.test_sample_id = int(indices['test'][0])
        if st.button('Use Random Unseen Test Sample', type='primary'):
            st.session_state.test_sample_id = int(np.random.default_rng().choice(indices['test']))
        sample_id = st.selectbox('Unseen test sample ID', options=[int(i) for i in indices['test']], key='test_sample_id')
        row = frame.iloc[sample_id]
        features = row[FEATURES].to_numpy(dtype=float)
        result = dict(truth=row[TARGETS].to_numpy(dtype=float), features=features, prediction=predictor.predict(features))
        st.caption(f'Held-out test sample {sample_id}. Not used to train or select the model.')

    truth, features, prediction = result['truth'], result['features'], result['prediction']
    st.subheader('2. What the Model Sees')
    st.markdown('#### Physical Formation')
    st.caption('Blue solid aircraft = ground truth. Orange outlined aircraft = ANN prediction. Grey aircraft = follower.')
    for column,title,figure in zip(st.columns(3),['Top View','Side View','Roll'],
                                  [formation_view(truth,prediction),side_view(truth,prediction),roll_view(truth,prediction)]):
        with column:
            st.markdown(f'**{title}**')
            st.plotly_chart(figure,width='stretch',key=f'physical_{title}')
    st.markdown('#### Follower Camera Observation')
    left,right = st.columns(2)
    full_frame = left.checkbox('Show full camera frame',value=False)
    show_missing = right.checkbox('Show missing keypoints as a ground-truth overlay',value=False)
    st.caption('Automatic zoom with padding; the dotted cross marks the camera center. '
               'Enable the full frame to compare apparent size and location at a fixed pixel scale.')
    st.plotly_chart(camera_view(features,config,truth,show_missing,full_frame),width='stretch',key='camera_plot')
    st.write(f"**Detected keypoints: {int(features.reshape(14,3)[:,2].sum())} / 14**")
    if show_missing:
        st.caption('Missing crosses are ideal ground-truth locations for illustration; they are not ANN inputs.')
    st.code('14 Camera Keypoints → 42 ML Features → ANN → X, Y, Z, Roll', language=None)
    with st.expander('View the 42 ML input features'):
        st.dataframe(pd.DataFrame(features.reshape(14,3), columns=['Normalized x','Normalized y','Visible'],
                                 index=pd.Index(range(1,15), name='Keypoint')), width='stretch')

    st.subheader('3. Model Output')
    difference = np.abs(prediction-truth)
    st.table(pd.DataFrame({'Parameter':['Distance X','Lateral Y','Vertical Z','Roll'],
                          'Ground Truth':human_pose(truth,2),'ML Prediction':human_pose(prediction,2),
                          'Difference':[f'{v:.2f} {unit}' for v,unit in zip(difference,['m','m','m','°'])]}).set_index('Parameter'))
    left, right = st.columns(2)
    left.metric('3D Position Error', f'{np.linalg.norm(prediction[:3]-truth[:3]):.2f} m')
    right.metric('Roll Error', f'{difference[3]:.2f}°')
    st.caption('True pose creates the scene and supplies comparison labels. Only 42 camera features enter the ANN.')

elif page == 'Model Performance':
    st.subheader('The trained ANN')
    st.code('42 → 100 + ReLU → 100 + ReLU → 4', language=None)
    st.write(f"Adam · learning rate {training['learning_rate']} · batch size {training['batch_size']} · "
             f"{training['epochs_completed']} epochs · best epoch {training['best_epoch']}")
    st.write(f"Train / validation / test: {len(indices['train']):,} / {len(indices['validation']):,} / {len(indices['test']):,}")
    st.subheader('Training vs Validation Loss')
    st.plotly_chart(loss_view(pd.read_csv(ROOT/'results/training_history.csv')), width='stretch', key='loss_plot')
    st.caption('Squared prediction differences on standardized targets. Validation observations do not update weights.')
    st.subheader('Held-out regression performance')
    scores = held_out_scores(stamp('models/best_model.pt'), stamp('models/target_scaler.joblib'),
                             stamp('data/raw/formation_flying_synthetic.csv'), stamp('data/processed/split_indices.npz'))
    for column, name in zip(st.columns(4), ['x','y','z','roll']):
        column.metric(f'{name.upper()} R²', f'{scores[name]:.4f}')
    st.write('Because pose estimation is formulated as regression, conventional classification accuracy is not applicable.')
    st.caption(f'Computed from saved-model predictions on all {len(indices["test"]):,} held-out test rows. '
               'R² near 1 indicates good agreement; 0 corresponds to predicting the test-target mean.')

else:
    st.subheader('How It Works')
    steps = [('Synthetic geometry', 'Fourteen fixed, identified structural points describe one aircraft.'),
             ('Random pose generation', 'Uniform candidate poses define X, Y, Z and roll in supported ranges.'),
             ('Pinhole projection', 'Rotate about forward X, translate, then project into the follower camera.'),
             ('Noise and occlusion', 'Add Gaussian pixel noise, drop random features, and check image bounds.'),
             ('42-feature representation', 'Each point supplies normalized u, normalized v and visibility. Missing triples are zero.'),
             ('Train / validation / test split', '70% train, 15% validation and 15% held-out test, using seed 42.'),
             ('ANN training', 'Fit the target scaler on training data only; use Adam and validation-selected saving.'),
             ('Prediction', 'The saved ANN maps camera features to four outputs; inverse scaling restores metres and degrees.')]
    for i, (title, text) in enumerate(steps, 1):
        st.markdown(f'**{i}. {title}** — {text}')
    st.subheader('Dataset settings')
    st.write(f'{len(frame):,} synthetic observations · 42 inputs · 4 regression outputs · seed 42')
    st.write(f"Train / validation / test: {len(indices['train']):,} / {len(indices['validation']):,} / {len(indices['test']):,}")
    st.write(f"Camera: {config['width']} × {config['height']} px · fx = fy = {config['fx']:g} px · center (640, 360)")
    st.write(f"Noise: σ = {config['noise_std_px']:g} px · missingness: {100*config['occlusion_probability']:g}% per point")
    st.write('Ranges: X 20–100 m · Y −20 to +20 m · Z −10 to +10 m · roll −45° to +45°')
    st.code('u = cx + fx · Yc / Xc\nv = cy − fy · Zc / Xc', language=None)
    st.caption('Forward +X, right +Y, upward +Z; image v increases downward. Occlusion is random missingness.')
    with st.expander('View the base 14-point aircraft geometry'):
        st.image(str(ROOT/'results/aircraft_keypoints.png'), width=700)
    with st.expander('View raw dataset sample'):
        st.dataframe(frame.head(), hide_index=True, width='stretch')
    st.caption('Inspired by the reference structural-feature ANN, adapted from classification to regression. '
               'Real-image detection, pitch/yaw, particle filtering, and flight control remain outside scope.')
