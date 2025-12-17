"""
Training Module - Train ML model ตาม config
"""
import yaml
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from data_loader import load_data, split_data


def load_config(path='../config/model_config.yaml'):
    """อ่าน config จากไฟล์ YAML"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"โหลด config จาก {path}")
    return config


def create_model(config):
    """สร้าง model ตาม config"""
    model_type = config['model']['type']
    
    if model_type == 'random_forest':
        params = config['model']['random_forest']
        model = RandomForestClassifier(**params)
    elif model_type == 'svm':
        params = config['model']['svm']
        model = SVC(**params)
    else:
        raise ValueError(f"ไม่รู้จัก model: {model_type}")
    
    print(f"สร้าง {model_type} model")
    return model


def main():
    print("=" * 50)
    print("เริ่มต้น Training Pipeline")
    print("=" * 50)
    
    # 1. โหลด config
    config = load_config()
    
    # 2. โหลดข้อมูล
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scale features เรียบร้อย")
    
    # 4. สร้างและ train model
    model = create_model(config)
    
    # Cross-validation
    if config['training']['cross_validation']:
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=config['training']['cv_folds']
        )
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train final model
    model.fit(X_train_scaled, y_train)
    
    # 5. ประเมินผล
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 6. บันทึก model
    model_data = {
        'model': model,
        'scaler': scaler,
        'config': config,
        'feature_names': feature_names,
        'target_names': list(target_names),
        'timestamp': datetime.now().isoformat()
    }
    joblib.dump(model_data, config['output']['model_path'])
    print(f"บันทึก model ที่ {config['output']['model_path']}")
    
    print("=" * 50)
    print("Training เสร็จสิ้น!")
    print("=" * 50)


if __name__ == "__main__":
    main()
