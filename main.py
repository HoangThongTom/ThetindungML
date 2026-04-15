# (phần của người 5)
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# Import phần xử lý dữ liệu (Người 2 + 3)
from Train.Preprocess import preprocess, train_test_split
from Train.Encoding import encode_features

# Import model (Người 1)
from GradientBoostingClassifier.models.gradient_boosting import GradientBoostingClassifierCustom

# Import toàn bộ metric từ Người 4
from GradientBoostingClassifier.metrics.accuracy import accuracy_score
from GradientBoostingClassifier.metrics.confusion_matrix import confusion_matrix
from GradientBoostingClassifier.metrics.classification_report import classification_report
from GradientBoostingClassifier.visualization.confusion_matrix_display import confusion_matrix_display


def main():
    print("=" * 55)
    print("  German Credit Risk — Gradient Boosting Classifier")
    print("=" * 55)

    # ── Load và chuẩn bị dữ liệu ─────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "Data", "raw_data.csv")
    df = pd.read_csv(data_path)

    X, y = preprocess(df, target_col="class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_enc, X_test_enc = encode_features(X_train, X_test)

    print(f"\nDữ liệu: {len(X_train)} train / {len(X_test)} test")

    # ── TRAIN: gọi fit() ─────────────────────────────────────────────
    print("\nĐang huấn luyện GradientBoostingClassifier...")

    model = GradientBoostingClassifierCustom(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
    )
    model.fit(X_train_enc, y_train.to_numpy())

    print("Huấn luyện xong.")

    # ── PREDICT: gọi predict() ───────────────────────────────────────
    y_pred    = model.predict(X_test_enc)
    y_test_np = y_test.to_numpy()

    # ── HIỂN THỊ METRIC: gọi toàn bộ metric từ Người 4 ──────────────
    print("\n" + "=" * 55)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 55)

    # 1. Accuracy
    acc = accuracy_score(y_test_np, y_pred)
    print(f"\nAccuracy : {acc:.4f}  ({acc * 100:.1f}%)")

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test_np, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Dự đoán: 0   Dự đoán: 1")
    print(f"  Thực tế: 0      {cm[0][0]:5d}        {cm[0][1]:5d}")
    print(f"  Thực tế: 1      {cm[1][0]:5d}        {cm[1][1]:5d}")

    # 3. Classification Report
    report = classification_report(y_test_np, y_pred)
    print("\nClassification Report:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1-score':>10} {'Support':>8}")
    print(f"  {'-' * 52}")
    for cls, m in report.items():
        print(f"  {str(cls):<12} {m['precision']:>10.4f} {m['recall']:>8.4f}"
              f" {m['f1_score']:>10.4f} {m['support']:>8}")

    # 4. Visualize Confusion Matrix
    print("\nĐang vẽ confusion matrix...")
    display = confusion_matrix_display(cm, display_labels=["bad (0)", "good (1)"])
    display.plot(
        title="Confusion Matrix — German Credit",
        save_path="confusion_matrix.png",
    )

    print("\n✓ Hoàn tất! File confusion_matrix.png đã được lưu.")


if __name__ == "__main__":
    main()# (phần của người 5)
