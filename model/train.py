# ============================================================
# train.py — Train the Fraud Detection ML Model
# ============================================================

# Step 1 — Import all libraries we need
import pandas as pd                          # for loading and handling data
import numpy as np                           # for numerical operations
import joblib                                # for saving the trained model
from sklearn.model_selection import train_test_split   # to split data
from sklearn.preprocessing import StandardScaler       # to scale Amount column
from sklearn.ensemble import RandomForestClassifier    # our ML model
from sklearn.metrics import (                          # to measure performance
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE     # to handle class imbalance

# ============================================================
# Step 2 — Load the dataset
# ============================================================
print("Loading dataset...")
df = pd.read_csv("data/raw/creditcard.csv")  # read the CSV file
print(f"Dataset shape: {df.shape}")          # print rows and columns
print(f"Fraud cases: {df['Class'].sum()}")   # print how many frauds exist

# ============================================================
# Step 3 — Prepare features and target
# ============================================================
# Drop 'Time' column — not useful for prediction
df = df.drop(columns=["Time"])

# X = all columns except Class (these are our inputs)
X = df.drop(columns=["Class"])

# y = Class column (0 = genuine, 1 = fraud) — this is what we predict
y = df["Class"]

# ============================================================
# Step 4 — Scale the Amount column
# ============================================================
scaler = StandardScaler()                         # create scaler object
X["Amount"] = scaler.fit_transform(              # scale Amount to mean=0, std=1
    X["Amount"].values.reshape(-1, 1)
)

# ============================================================
# Step 5 — Split into train and test sets (80% train, 20% test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% data for testing
    random_state=42,      # so results are reproducible
    stratify=y            # keep same fraud ratio in both splits
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# Step 6 — Handle class imbalance using SMOTE
# ============================================================
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)                    # create SMOTE object
X_train, y_train = smote.fit_resample(            # generate synthetic fraud samples
    X_train, y_train
)
print(f"After SMOTE — Training samples: {len(X_train)}")

# ============================================================
# Step 7 — Train the Random Forest model
# ============================================================
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,           # 100 decision trees
    class_weight="balanced",    # give more weight to fraud class
    random_state=42             # reproducible results
)
model.fit(X_train, y_train)     # train the model
print("Model trained successfully!")

# ============================================================
# Step 8 — Evaluate the model on test set
# ============================================================
print("\n===== MODEL EVALUATION =====")
y_pred = model.predict(X_test)              # predict on test data
y_prob = model.predict_proba(X_test)[:, 1] # get fraud probability scores

precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

# False Positive Rate = FP / (FP + TN)
TN, FP, FN, TP = cm.ravel()
fpr = FP / (FP + TN)

print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1-Score         : {f1:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nFull Report:\n{classification_report(y_test, y_pred)}")

# ============================================================
# Step 9 — Save model and scaler together
# ============================================================
joblib.dump(
    {"model": model, "scaler": scaler, "columns": list(X.columns)},
    "model/fraud_model.pkl"      # save as a single file
)
print("\nModel saved as model/fraud_model.pkl ✅")