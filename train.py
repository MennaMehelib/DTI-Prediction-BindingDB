import time
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from models import (
    get_sklearn_model, FCNN, MHAFCNN,
    train_torch_model, predict_torch
)
from evaluation import compute_metrics
from data_loader import train_test_split_stratified
from config import MODEL_CONFIG


# SINGLE MODEL TRAINING

def train_sklearn_model(model_name, X_train, y_train, X_test, y_test):

    model = get_sklearn_model(model_name)
    
    # Training
    t_start = time.time()
    model.fit(X_train, y_train)
    train_t = time.time() - t_start
    
    # Prediction
    t_pred = time.time()
    y_pred = model.predict(X_test)
    pred_t = time.time() - t_pred
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'train_time': round(train_t, 3),
        'pred_time': round(pred_t, 3)
    }


def train_dl_model(model_class, X_train, y_train, X_test, y_test, input_dim):
   
    model = model_class(input_dim)
    
    # Training
    t_start = time.time()
    model = train_torch_model(
        model, X_train, y_train,
        epochs=MODEL_CONFIG['dl_epochs'],
        batch_size=MODEL_CONFIG['dl_batch_size'],
        lr=MODEL_CONFIG['dl_learning_rate']
    )
    train_t = time.time() - t_start
    
    # Prediction
    t_pred = time.time()
    y_pred, y_prob = predict_torch(model, X_test)
    pred_t = time.time() - t_pred
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'train_time': round(train_t, 3),
        'pred_time': round(pred_t, 3)
    }



# FULL EXPERIMENT PIPELINE

def run_experiment_for_threshold(Th, X, y, feature_extractor=None):

    print(f"\n  {'─'*55}")
    print(f"Threshold = {Th}")
    print(f"  {'─'*55}")
    
    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(
        X, y, test_size=MODEL_CONFIG['test_size']
    )
    print(f"    Train: {X_tr.shape}  |  Test: {X_te.shape}")
    
    records = []
    input_dim = X_tr.shape[1]
    
    # Train sklearn models
    sklearn_models = ['DTC', 'MLP', 'RFC']
    for model_name in sklearn_models:
        result = train_sklearn_model(model_name, X_tr, y_tr, X_te, y_te)
        metrics = compute_metrics(y_te, result['predictions'], result['probabilities'])
        
        metrics.update({
            'Threshold': Th,
            'Model': model_name,
            'Train_Time': result['train_time'],
            'Pred_Time': result['pred_time'],
            'Total_Time': result['train_time'] + result['pred_time']
        })
        records.append(metrics)
        
        print(f"    {model_name:10s} → Acc={metrics['Accuracy']:.2f}%  "
              f"F1={metrics['F1-Score']:.2f}%  AUC={metrics['ROC-AUC']:.2f}%  "
              f"[train={result['train_time']:.2f}s]")
    
    # Train deep learning models
    dl_models = [('FCNN', FCNN), ('MHA-FCNN', MHAFCNN)]
    for dl_name, ModelCls in dl_models:
        result = train_dl_model(ModelCls, X_tr, y_tr, X_te, y_te, input_dim)
        metrics = compute_metrics(y_te, result['predictions'], result['probabilities'])
        
        metrics.update({
            'Threshold': Th,
            'Model': dl_name,
            'Train_Time': result['train_time'],
            'Pred_Time': result['pred_time'],
            'Total_Time': result['train_time'] + result['pred_time']
        })
        records.append(metrics)
        
        print(f"    {dl_name:10s} → Acc={metrics['Accuracy']:.2f}%  "
              f"F1={metrics['F1-Score']:.2f}%  AUC={metrics['ROC-AUC']:.2f}%  "
              f"[train={result['train_time']:.2f}s]")
    
    return records


# COMPLEXITY ANALYSIS

def run_complexity_analysis(X_train, y_train, X_test, y_test):

    print("\n" + "=" * 65)
    print("COMPLEXITY ANALYSIS")
    print("=" * 65)
    
    cx_rows = []
    
    # Sklearn models
    for model_name in ['DTC', 'MLP', 'RFC']:
        result = train_sklearn_model(model_name, X_train, y_train, X_test, y_test)
        cx_rows.append({
            'Model': model_name,
            'Train_s': result['train_time'],
            'Pred_s': result['pred_time'],
            'Total_s': result['train_time'] + result['pred_time']
        })
    
    # Deep learning models
    input_dim = X_train.shape[1]
    for dl_name, ModelCls in [('FCNN', FCNN), ('MHA-FCNN', MHAFCNN)]:
        result = train_dl_model(ModelCls, X_train, y_train, X_test, y_test, input_dim)
        cx_rows.append({
            'Model': dl_name,
            'Train_s': result['train_time'],
            'Pred_s': result['pred_time'],
            'Total_s': result['train_time'] + result['pred_time']
        })
    
    return pd.DataFrame(cx_rows)


# CLASSIFICATION REPORTS

def generate_classification_reports(y_test, predictions_dict):

    report_rows = []
    
    for name, preds in predictions_dict.items():
        rpt = classification_report(y_test, preds, output_dict=True, zero_division=0)
        for cls_key, cls_label in [('0', 'No'), ('1', 'Yes')]:
            report_rows.append({
                'Model': name,
                'Class': cls_label,
                'Precision': round(rpt[cls_key]['precision'] * 100, 2),
                'Recall': round(rpt[cls_key]['recall'] * 100, 2),
                'F1-Score': round(rpt[cls_key]['f1-score'] * 100, 2),
            })
    
    return pd.DataFrame(report_rows)