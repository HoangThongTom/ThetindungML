import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.ensemble import GradientBoostingClassifier

def main():
    # 1. Tải dữ liệu ung thư vú
    print("Đang tải dữ liệu...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 2. Chia tập dữ liệu (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Chuẩn hóa dữ liệu 
    # (Thực tế các mô hình dạng Tree không quá nhạy cảm với scale, 
    # nhưng slàm theo đúng yêu cầu sử dụng scaler của sklearn)
    print("Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Khởi tạo mô hình 
    # Set n_estimators nhỏ (10) và max_depth nhỏ (2) để tránh việc phải đợi quá lâu
    print("Đang khởi tạo Gradient Boosting (Custom)...")
    clf = GradientBoostingClassifier(
        n_estimators=10, 
        learning_rate=0.1, 
        max_depth=2
    )

    # 5. Huấn luyện mô hình
    print("Bắt đầu quá trình huấn luyện (vui lòng đợi một lát do thuật toán chia node đang dùng vòng lặp Python)...")
    start_time = time.time()
    
    clf.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"\n[+] Huấn luyện hoàn tất trong {train_time:.2f} giây.")

    # 6. Dự đoán trên tập Test
    print("Đang dự đoán trên tập kiểm tra...")
    y_pred = clf.predict(X_test_scaled)

    # 7. Đánh giá và báo cáo bằng sklearn
    print("\n" + "="*40)
    print("         KẾT QUẢ ĐÁNH GIÁ (TEST SET)")
    print("="*40)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

if __name__ == "__main__":
    main()