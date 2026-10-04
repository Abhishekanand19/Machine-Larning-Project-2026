"""Execute eight custom scenarios, all four pages, test controls and Plotly charts."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
try:
    import streamlit
except ImportError:
    sys.path.insert(0,str(root/'.local-deps'))
from streamlit.testing.v1 import AppTest
import numpy as np
import json
import hashlib

app = AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception, app.exception
assert app.radio[0].value == 'Try the Model'
model_hash = hashlib.sha256((root/'models/best_model.pt').read_bytes()).hexdigest()
protected = ['models/best_model.pt','models/target_scaler.joblib','data/raw/formation_flying_synthetic.csv']
artifact_hashes = {file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in protected}
labels = ['Distance ahead (X)','Left / Right offset (Y)','Height difference (Z)','Aircraft bank / roll']
scenarios = [[25,0,0,0],[90,0,0,0],[60,-15,0,0],[60,15,0,0],
             [60,0,-8,0],[60,0,8,0],[60,0,0,-40],[60,0,0,40]]
records = []
for pose in scenarios:
    for label,value in zip(labels,pose):
        next(slider for slider in app.slider if slider.label == label).set_value(float(value))
    next(button for button in app.button if button.label=='Generate Camera View & Predict').click().run()
    assert not app.exception, app.exception
    assert len(app.get('plotly_chart')) == 4
    assert len(app.metric) == 2
    result = app.session_state['custom_result']
    np.testing.assert_array_equal(result['truth'],pose)
    assert result['features'].shape == (42,)
    records.append(dict(truth=pose,prediction=result['prediction'].tolist(),seed=result['seed'],
                        visible_keypoints=int(result['features'].reshape(14,3)[:,2].sum())))
for low,high,axis in [(0,1,0),(2,3,1),(4,5,2),(6,7,3)]:
    assert records[high]['prediction'][axis] > records[low]['prediction'][axis]
previous = app.session_state['custom_result']['truth'].copy()
presets = {'Close & Level':[25,0,0,0],'Far & Level':[90,0,0,0],
           'Left & Above':[60,-15,8,0],'Right & Below':[60,15,-8,0],'Banked Formation':[60,8,3,40]}
for name,pose in presets.items():
    next(button for button in app.button if button.label==name).click().run()
    assert not app.exception, app.exception
    assert [next(slider for slider in app.slider if slider.label==label).value for label in labels]==pose
    np.testing.assert_array_equal(app.session_state['custom_result']['truth'],previous)
next(button for button in app.button if button.label=='Generate Camera View & Predict').click().run()
np.testing.assert_array_equal(app.session_state['custom_result']['truth'],presets['Banked Formation'])
app.checkbox[0].check().run()  # Full-frame camera toggle.
app.checkbox[1].check().run()  # Optional missing-point truth overlay.
assert not app.exception, app.exception
app.radio[1].set_value('Unseen Test Sample').run()
assert not app.exception, app.exception
test_ids = set(np.load(root/'data/processed/split_indices.npz')['test'].tolist())
app.selectbox[0].select(int(app.selectbox[0].options[1])).run()
assert app.selectbox[0].value in test_ids
app.button[0].click().run()
assert not app.exception, app.exception
assert app.selectbox[0].value in test_ids
app.radio[0].set_value('Model Performance').run()
assert not app.exception, app.exception
assert len(app.get('plotly_chart')) == 1
scores = json.loads((root/'results/evaluation_summary.json').read_text())['r2']
assert [metric.value for metric in app.metric] == [f'{scores[name]:.4f}' for name in ('x','y','z','roll')]
for page in ['Home','How It Works']:
    app.radio[0].set_value(page).run()
    assert not app.exception, app.exception
assert hashlib.sha256((root/'models/best_model.pt').read_bytes()).hexdigest() == model_hash
assert artifact_hashes == {file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in protected}
(root/'results/custom_demo_verification.json').write_text(json.dumps(dict(
    scenarios=records,test_mode='passed: selector and random button only use held-out IDs',
    plotly='passed: top, side, roll, auto/full camera and dynamic training charts',r2=scores,
    presets='all five populate controls without altering submitted scenario until Generate',
    protected_artifacts=artifact_hashes,model_sha256=model_hash,model_unchanged=True,
    dataset_and_scaler_unchanged=True,pages='all four passed without Python exceptions'),indent=2),encoding='utf-8')
print('App passed: eight custom scenarios, four pages, Plotly charts, test selectors and held-out R².')
