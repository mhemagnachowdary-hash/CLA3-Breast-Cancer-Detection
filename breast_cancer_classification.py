"""
CLA-3 Case Study: Breast Cancer Detection Using Wisconsin Diagnostic Data

Run:
    pip install -r requirements.txt
    python breast_cancer_classification.py

The script uses scikit-learn's copy of the Wisconsin Diagnostic dataset.
Class labels in scikit-learn: 0 = malignant, 1 = benign.
This is an educational classification exercise, not a medical diagnostic tool.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    print("Dataset shape:", X.shape)
    print("Class labels:", dict(enumerate(data.target_names)))
    print("Training and test split: 80% / 20%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (benign)": precision_score(y_test, y_pred, pos_label=1),
            "Recall (benign)": recall_score(y_test, y_pred, pos_label=1),
            "F1-score (benign)": f1_score(y_test, y_pred, pos_label=1),
            "Malignant recall": recall_score(y_test, y_pred, pos_label=0)
        })

        print(f"\n{name}")
        print(classification_report(
            y_test, y_pred, target_names=data.target_names, digits=4
        ))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=data.target_names,
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(name)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180)
    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved results in: {OUTPUT_DIR.resolve()}")
    print("Note: This model is for academic learning only, not clinical diagnosis.")

if __name__ == "__main__":
    main()
