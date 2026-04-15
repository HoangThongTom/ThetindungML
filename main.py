import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preprocessing (Người 2 + 3)
from Preprocessing.Preprocess import preprocess, train_test_split
from Preprocessing.endcoding import encode_features

# Model (Người 1)
from GradientBoostingClassifier.models.gradient_boosting import GradientBoostingClassifier

# Metrics (Người 4)
from GradientBoostingClassifier.metrics.accuracy import accuracy_score
from GradientBoostingClassifier.metrics.confusion_matrix import confusion_matrix
from GradientBoostingClassifier.metrics.classification_report import classification_report
from GradientBoostingClassifier.visualization.confusion_matrix_display import confusion_matrix_display


def main():
    print("=" * 55)
    print("  German Credit Risk — Gradient Boosting Classifier")
    print("=" * 55)

    #  Load dữ liệu 
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "raw_data.csv")
    df = pd.read_csv(data_path)
    print(f"\nĐã load dữ liệu: {df.shape[0]} mẫu, {df.shape[1]} cột")

    # ── Preprocess ────────────────────────────────────────────
    X, y = preprocess(df, target_col="class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_enc, X_test_enc = encode_features(X_train, X_test)

    print(f"Train: {len(X_train)} mẫu  |  Test: {len(X_test)} mẫu")

    # Train
    print("\nĐang huấn luyện GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
    )
    model.fit(X_train_enc, y_train.to_numpy())
    print("Huấn luyện xong.")

    #  Predict 
    y_pred    = model.predict(X_test_enc)
    y_test_np = y_test.to_numpy()

    #  Metrics
    print("\n" + "=" * 55)
    print("  Kết quả đánh giá")
    print("=" * 55)

    # 1. Accuracy
    acc = accuracy_score(y_test_np, y_pred)
    print(f"\nAccuracy : {acc:.4f}  ({acc * 100:.1f}%)")

    # 2. Confusion Matrix
    cm, labels = confusion_matrix(y_test_np, y_pred)
    label_names = {0: "bad", 1: "good"}
    named = [label_names.get(l, str(l)) for l in labels]

    print("\nConfusion Matrix:")
    print(f"                 Dự đoán: {named[0]:<6}  Dự đoán: {named[1]}")
    print(f"  Thực tế: {named[0]:<6}      {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"  Thực tế: {named[1]:<6}      {cm[1][0]:>5}           {cm[1][1]:>5}")

    # 3. Classification Report
    report = classification_report(y_test_np, y_pred)
    print("\nClassification Report:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1-score':>10} {'Support':>8}")
    print(f"  {'-' * 50}")
    for cls, m in report.items():
        cls_name = label_names.get(cls, str(cls))
        print(f"  {cls_name:<10} {m['precision']:>10.4f} {m['recall']:>8.4f}"
              f" {m['f1_score']:>10.4f} {m['support']:>8}")

    # 4. Visualize Confusion Matrix
    print("\nĐang vẽ confusion matrix...")
    display = confusion_matrix_display(cm, display_labels=named)
    display.plot(
        title="Confusion Matrix — German Credit",
        save_path="confusion_matrix.png",
    )

    print("\n Hoàn tất! File confusion_matrix.png đã được lưu.")


if __name__ == "__main__":
    main()
