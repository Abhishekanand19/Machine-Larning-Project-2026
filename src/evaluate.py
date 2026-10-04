import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils import ROOT, FEATURES, TARGETS, save_json
from preprocessing import load_splits
from predict import PosePredictor

def regression_scores(actual, predicted):
    """Per-output coefficient of determination, not classification accuracy."""
    return dict(zip(('x','y','z','roll'),r2_score(actual,predicted,multioutput='raw_values').tolist()))

def main():
    frame,indices,_ = load_splits()
    test = frame.iloc[indices['test']]
    actual = test[TARGETS].to_numpy()
    predicted = PosePredictor().predict(test[FEATURES].to_numpy())
    difference = np.abs(predicted-actual)
    output = pd.DataFrame({'sample_id':indices['test']})
    correlation = {}
    for i, name in enumerate(('x','y','z','roll')):
        output[f'true_{name}'] = actual[:,i]
        output[f'predicted_{name}'] = predicted[:,i]
        output[f'difference_{name}'] = difference[:,i]
        correlation[name] = float(np.corrcoef(actual[:,i],predicted[:,i])[0,1])
        unit = 'degrees' if name == 'roll' else 'm'
        fig,ax = plt.subplots(figsize=(5,5))
        ax.scatter(actual[:,i],predicted[:,i],s=6,alpha=.3,color='#156082')
        lo,hi = min(actual[:,i].min(),predicted[:,i].min()),max(actual[:,i].max(),predicted[:,i].max())
        ax.plot([lo,hi],[lo,hi],'--',color='#a34f38',label='Ideal prediction')
        ax.set(xlabel=f'Actual {name.upper()} ({unit})',ylabel=f'Predicted {name.upper()} ({unit})',
               title=f'Held-out test set: {name.upper()}')
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(ROOT/f'results/predicted_vs_actual_{name}.png',dpi=160)
        plt.close(fig)
    output['position_error_m'] = np.linalg.norm(predicted[:,:3]-actual[:,:3],axis=1)
    output['roll_error_deg'] = difference[:,3]
    output.to_csv(ROOT/'results/test_predictions.csv',index=False)
    output.iloc[np.random.default_rng(42).choice(len(output),10,replace=False)].to_csv(ROOT/'results/sample_predictions.csv',index=False)
    summary = dict(test_samples=len(test),correlation=correlation,r2=regression_scores(actual,predicted),
                   position_error_median_m=float(output.position_error_m.median()),
                   position_error_p90_m=float(output.position_error_m.quantile(.9)),
                   roll_error_median_deg=float(output.roll_error_deg.median()),
                   roll_error_p90_deg=float(output.roll_error_deg.quantile(.9)),
                   predicted_standard_deviation=predicted.std(axis=0).tolist(),
                   actual_standard_deviation=actual.std(axis=0).tolist())
    save_json(ROOT/'results/evaluation_summary.json',summary)
    print(summary)

if __name__ == '__main__':
    main()
