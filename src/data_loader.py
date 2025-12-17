"""
Data Loader Module - โหลดและเตรียมข้อมูล Iris
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    """โหลด Iris dataset"""
    iris = load_iris()
    print(f"โหลดข้อมูลสำเร็จ: {iris.data.shape[0]} ตัวอย่าง, {iris.data.shape[1]} features")
    return iris.data, iris.target, iris.feature_names, iris.target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """แบ่งข้อมูลเป็น train และ test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"แบ่งข้อมูล: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


# ทดสอบเมื่อรันไฟล์นี้โดยตรง
if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Features: {feature_names}")
    print(f"Classes: {list(target_names)}")
