"""
Evaluation Module - ประเมินผล ML model
"""
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_data, split_data


def load_model(path):
    """โหลด model จากไฟล์"""
    model_data = joblib.load(path)
    print(f"โหลด model จาก {path}")
    print(f"  - ประเภท: {type(model_data['model']).__name__}")
    print(f"  - Train เมื่อ: {model_data['timestamp']}")
    return model_data


def evaluate(model, X_test, y_test, target_names):
    """ประเมินผล model"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return metrics, y_pred


def plot_confusion_matrix(cm, target_names, save_path):
    """สร้าง confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # แสดงตัวเลขในแต่ละช่อง
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึก confusion matrix ที่ {save_path}")


def main():
    print("=" * 50)
    print("เริ่มต้น Evaluation Pipeline")
    print("=" * 50)
    
    # 1. โหลด model
    model_data = load_model('../models/model.joblib')
    model = model_data['model']
    scaler = model_data['scaler']
    config = model_data['config']
    target_names = model_data['target_names']
    
    # 2. โหลดข้อมูล (ใช้ random_state เดียวกับตอน train)
    X, y, _, _ = load_data()
    _, X_test, _, y_test = split_data(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # 3. Scale ข้อมูล
    X_test_scaled = scaler.transform(X_test)
    
    # 4. ประเมินผล
    metrics, y_pred = evaluate(model, X_test_scaled, y_test, target_names)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    
    # 5. สร้าง confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, target_names, '../results/confusion_matrix.png')
    
    # 6. บันทึกผลลัพธ์
    results = {
        'accuracy': metrics['accuracy'],
        'model_type': config['model']['type'],
        'timestamp': datetime.now().isoformat()
    }
    with open('../results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"บันทึกผลลัพธ์ที่ ../results/metrics.json")
    
    print("=" * 50)
    print("Evaluation เสร็จสิ้น!")
    print("=" * 50)


if __name__ == "__main__":
    main()
