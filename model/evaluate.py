import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop customerID (not useful for prediction)
data = data.drop("customerID", axis=1)

# Encode target variable
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical variables (Label Encoding for simplicity)
cat_cols = data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split (same as training script)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the saved model
model = joblib.load("model/churn_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Extra: Show classification report (precision, recall, f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
