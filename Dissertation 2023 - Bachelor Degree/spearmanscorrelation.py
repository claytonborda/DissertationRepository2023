import numpy as np
import mord

# Data
data = np.array([
    [12047, 4487, 6676],  # Positive tweets
    [10452, 3470, 7110]   # Negative tweets
])

# Calculate the ratio of positive to negative tweets
ratio = data[0, :] / data[1, :]

# Match outcomes
match_outcomes = ["Won", "Drawn", "Lost"]

# Indices representing the match outcomes
match_outcome_indices = np.arange(len(match_outcomes))

# Reshape data for the regression model
X = ratio.reshape(-1, 1)
y = match_outcome_indices

# Fit an ordinal logistic regression model
model = mord.LogisticAT(alpha=0)
model.fit(X, y)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Predict match outcomes based on the ratio of positive to negative tweets
predicted_outcomes = model.predict(X)
print(f"Predicted Match Outcomes: {predicted_outcomes}")
