import pandas as pd

# =========================
# IMPORTS
# =========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =========================
# COMMON FUNCTION
# =========================
def run_models(df, target_column, dataset_name):
    print(f"\n--- {dataset_name} Dataset ---")

    # Split features & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # If target is string (like M/B), convert to numbers
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    best_model = None
    best_accuracy = 0

    # Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.2f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = name

    print(f"Best Model: {best_model}")


# =========================
# 1️⃣ DIABETES DATASET
# =========================
diabetes_df = pd.read_csv("diabetes.csv")
run_models(diabetes_df, "Outcome", "Diabetes")


# =========================
# 2️⃣ HEART DISEASE DATASET
# =========================
heart_df = pd.read_csv("heart.csv")
run_models(heart_df, "target", "Heart Disease")


# =========================
# 3️⃣ BREAST CANCER DATASET
# =========================
cancer_df = pd.read_csv("breast_cancer.csv")

# Clean unnecessary columns (Kaggle dataset)
cancer_df = cancer_df.drop(['id', 'Unnamed: 32'], axis=1)

run_models(cancer_df, "diagnosis", "Breast Cancer")