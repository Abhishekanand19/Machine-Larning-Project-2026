# Two-minute demonstration

## 0:00–0:20 — Overview

“Our project is vision-based relative pose estimation for formation flying. The follower observes a leader aircraft with a camera. We predict forward distance, lateral displacement, vertical displacement, and roll.”

“This is a synthetic machine learning experiment. We are not controlling an aircraft.”

## 0:20–0:40 — Dataset and keypoints

Open **How It Works** and expand the base geometry if useful.

“We fixed fourteen structural points on a simple aircraft and generated twenty thousand random poses. A pinhole camera projects these points into image coordinates. We add two-pixel Gaussian noise and about ten percent random missing features.”

“Each point gives two normalized coordinates and a visibility flag. Fourteen times three gives forty-two inputs.”

## 0:40–0:55 — ANN

Open **Model Performance**.

“Our PyTorch network has two hidden layers of one hundred neurons each, with ReLU. Adam trains it to predict four continuous outputs. The reference paper uses classification; our adaptation uses regression.”

## 0:55–1:30 — Unseen test prediction

Open **Try the Model**. Choose **Custom Flight Scenario**, set X, Y, Z and roll, then click **Generate Camera View & Predict**.

“These sliders define the true physical scenario. The simulator projects it into keypoints with noise and missingness. The top and side views show the true and predicted formation. The roll indicator shows aircraft bank. The large camera view shows exactly the features supplied to the ANN.”

“The model receives only coordinates and visibility flags, never the slider values. Here is the true pose, and beside it is the prediction. These are their absolute differences.”

Point to the actual displayed values; do not promise a particular error.

“The position card is the distance between the true and predicted 3D positions. The roll card is their angular difference.”

Increase X and regenerate: the aircraft looks smaller. Change roll and regenerate: the projected structure rotates. If time allows, switch to **Unseen Test Sample** and click **Use Random Unseen Test Sample** to show an observation excluded from training.

## 1:30–1:50 — Learning evidence

Return to **Model Performance**.

“Both training and validation losses fall. We selected epoch two hundred ninety-nine using validation data. We reserved three thousand test observations for final evaluation.”

“We report R² on those held-out test observations. All four scores are close to one. Classification accuracy does not apply to our regression model.”

## 1:50–2:00 — Conclusion

“This demonstrates that a simple ANN can learn pose from synthetic structural features. Real images, varying aircraft geometry, pitch and yaw, and flight control are outside our current scope.”

Before presenting: start Streamlit, check all pages, and fill in team details. Use Close & Level and Far & Level to compare image size; use Show full camera frame for a fixed pixel scale. Presets populate controls, but the displayed scenario changes only after Generate. The default observation seed makes comparisons repeatable. If a scenario has a larger error, explain noise, missing features, and synthetic-model limitations honestly.
