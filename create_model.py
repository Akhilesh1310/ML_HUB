# create_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train a basic model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model to a file named "model.pkl"
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as test_model.pkl")
