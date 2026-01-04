# pl-predictive-model-RFC

EPL Match Prediction via Random Forest Classification

This repository moves beyond manual weights to implement a Random Forest Classifier (RFC). By training on over 500 Premier League matches across two seasons, the model learns the non-linear relationships between team stats and the final result.

How it Works:
The model consists of an ensemble of 200 Decision Trees. Each tree analyzes a random subset of features (like Goal Difference, Red Cards, or Rest Days) to vote on the most likely outcome.

Data Science Highlights:
- Feature Engineering: Combines 14 distinct features per match, including defense ratings and efficiency metrics.

- Model Persistence: Uses joblib to save the trained "brain" (.pkl), allowing for instant inference without retraining.

- Explainable AI: Includes scripts to visualize Feature Importance, showing which stats actually drive wins in the Premier League.

- Multi-Season Training: Handles the "Promoted Team Problem" by using league-average baselines for teams newly arrived from the Championship.

Visualizing the Brain:
The project generates a feature importance graph to show which metrics the model "valued" most during training.

Tech Stack:
Python

Scikit-Learn (Random Forest, Label Encoding)

Pandas (Data manipulation)

Matplotlib (Visualization)

Joblib (Model Serialization)
