import cloudpickle
import joblib
import os

# Assuming your trained model is called `model` and `X` is your training DataFrame
from your_training_script import model, X  # <-- replace with actual imports if needed

# Ensure the models folder exists
os.makedirs('models', exist_ok=True)

# Save the model using cloudpickle
with open('models/house_price_model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)

# Save the feature names using joblib
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
