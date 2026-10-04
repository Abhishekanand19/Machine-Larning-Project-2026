# Vision-Based Relative Pose Estimation for Autonomous Formation Flying

**Team:** [Member 1 name / USN], [Member 2 name / USN]

## Problem Statement

A follower aircraft observing a leader needs relative position and orientation information to understand their formation. This mini-project estimates four quantities from a synthetic camera observation: forward distance X, lateral displacement Y, vertical displacement Z, and relative roll. It demonstrates machine learning for pose estimation; it does not implement aircraft control.

## Dataset Details

We generated 20,000 independent synthetic observations with random seed 42. A fixed, symmetric aircraft consists of 14 named structural keypoints. Candidate poses are sampled uniformly: X 20–100 m, Y −20 to +20 m, Z −10 to +10 m, and roll −45° to +45°. A fixed 1280 × 720 px pinhole camera uses focal lengths 400 px and principal point (640,360), with forward +X, right +Y, and upward +Z.

Each sample includes 42 inputs: the two normalized image coordinates and visibility flag of each of 14 points. Four continuous pose targets complete the 46-column CSV. Gaussian noise has standard deviation 2 px and independent missingness probability is 10% per point. Off-frame and nonpositive-depth points are invalid. Samples require at least eight visible points. Two of 20,002 candidates were rejected. Validation found no null values, non-finite values, or duplicate rows; the observed missing-feature fraction was 9.9932%.

## Approach

The reference paper by Rohan Punnoose uses structural features and an ANN for coarse pose classification, followed by a particle filter. We retain its 14-point, 42-input concept and two 100-neuron hidden layers, but replace discrete classification with continuous multi-output regression. We do not reproduce its complete pipeline or implement a real-image feature detector.

For each point, roll rotation about +X precedes translation. Camera projection is `u = 640 + 400*Yc/Xc`, `v = 360 − 400*Zc/Xc`. Coordinates are normalized to [-1,1]; an invisible point is `(0,0,0)`. Training-only target standardization balances the four outputs. Inputs are not scaled again.

## Implementation

Python scripts generate and validate data, persist split indices, train a PyTorch ANN, evaluate the saved model, and perform reusable inference. Seed-42 splitting yields 14,000 training, 3,000 validation, and 3,000 test samples. The test set is excluded from training and model selection.

The network is `42 → 100 + ReLU → 100 + ReLU → 4`, with 14,804 parameters. Training uses Adam, learning rate 0.001, batch size 256, standardized-target MSE loss, maximum 300 epochs, and early-stopping patience 40. The executed run completed 300 epochs and selected epoch 299 by lowest validation loss. The checkpoint and target scaler are saved under `models/`.

A four-page Streamlit application opens on Try the Model. Custom sliders define a true synthetic pose; the same simulator used for dataset generation converts it into noisy, visibility-aware camera observations. Only 42 features reach the saved prediction class. Interactive Plotly views show formation geometry and the follower camera. The demo compares true and predicted pose, absolute differences, single-sample 3D position error, and roll error. A secondary mode selects exclusively held-out test samples. Model Performance renders a live training curve and held-out R²; How It Works holds technical details.

## Results

Validation loss decreased from 0.488323 at epoch 1 to a best value of 0.00755544 at epoch 299. On the 3,000-sample test set, actual-versus-predicted Pearson correlations were X 0.9928, Y 0.9984, Z 0.9983, and roll 0.9955. Median single-sample 3D position error was 1.5655 m and median absolute roll error was 1.4031°. Their 90th percentiles were 4.4958 m and 3.8048°. These measured synthetic results indicate successful pose learning, with some outliers visible in the scatter plots.

Test-set R² values are X 0.985520, Y 0.996718, Z 0.996571, and roll 0.990956; conventional classification accuracy does not apply. Saved evidence includes six dataset sample images, base geometry, training and prediction plots, test prediction CSVs, and reports. Nine integration tests passed. Streamlit execution tests passed for eight custom scenarios spanning distance, displacement and roll changes, all pages, Plotly figures, and both test selectors. The simulator refactor preserves the generated data and requires no retraining.

## Conclusion

A compact ANN can learn relative pose from identified structural keypoints in this controlled synthetic setting. The main contributions are reproducible generation, mathematically consistent projection, visibility-aware inputs, training without test leakage, and a clear live demonstration. Accuracy depends on fixed known geometry, camera calibration, and feature identities. Pitch/yaw, real image detection, physical self-occlusion, dynamics, uncertainty, and flight control remain outside scope. Future work should test new geometries/cameras, stronger observation corruption, and labelled real keypoints before making real-world claims.

**Submission note:** Adapt this draft to the faculty's required PDF length. The supplied guidelines use a one-page heading and a two-page body instruction; slides and the final submission PDF are separate deliverables still to prepare.
