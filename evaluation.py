import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error
)
from scipy.stats import friedmanchisquare

from config import OUTPUT_DIR


# METRICS COMPUTATION

def compute_metrics(y_true, y_pred, y_prob=None):

    # Basic metrics
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    sens = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100
    
    # Specificity for binary classification
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0
    
    # ROC-AUC
    if y_prob is None:
        y_prob = y_pred
    roc_auc = roc_auc_score(y_true, y_prob) * 100
    
    # Error metrics
    mae = mean_absolute_error(y_true, y_pred) * 100
    mse = mean_squared_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    
    return {
        'Accuracy': round(acc, 2),
        'Precision': round(prec, 2),
        'Sensitivity': round(sens, 2),
        'Specificity': round(spec, 2),
        'F1-Score': round(f1, 2),
        'Kappa': round(kappa, 2),
        'MCC': round(mcc, 2),
        'ROC-AUC': round(roc_auc, 2),
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
    }


# CONFUSION MATRIX VISUALIZATION

def plot_confusion_matrices(y_test, predictions_dict, model_order=None, title=""):

    if model_order is None:
        model_order = list(predictions_dict.keys())
    
    n_models = len(model_order)
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models+2, 4.5))
    
    for ax, name in zip(axes, model_order):
        cm_v = confusion_matrix(y_test, predictions_dict[name])
        sns.heatmap(cm_v, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No (0)', 'Yes (1)'],
                    yticklabels=['No (0)', 'Yes (1)'],
                    linewidths=0.5, cbar=True, cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 11})
        star = " ★" if name == 'RFC' else ""
        ax.set_title(f'({name}){star}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted label', fontsize=9)
        ax.set_ylabel('True label', fontsize=9)
    
    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_confusion_matrices_GAN.png", dpi=150, bbox_inches='tight')
    plt.close()


# PERFORMANCE VISUALIZATIONS

def plot_model_comparison(results_df, threshold=10, metrics=None):

    if metrics is None:
        metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
    
    models_plot = ['DTC', 'MLP', 'RFC', 'FCNN', 'MHA-FCNN']
    palette = ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7', '#c4ad66']
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    
    for ax, metric in zip(axes, metrics):
        sub_10 = results_df[results_df['Threshold'] == threshold]
        vals = []
        
        for m in models_plot:
            row = sub_10[sub_10['Model'] == m]
            vals.append(float(row[metric].values[0]) if not row.empty else 0)
        
        bars = ax.bar(models_plot, vals, color=palette, edgecolor='white', 
                      width=0.6, alpha=0.9)
        ax.set_ylim(min(v for v in vals if v > 0) * 0.975, 101.5)
        ax.set_title(f'{metric}\n(Th={threshold}, GAN balanced)', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric} (%)')
        ax.set_xticklabels(models_plot, rotation=25, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.25)
        
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.4,
                   f'{val:.2f}', ha='center', va='top', fontsize=9,
                   fontweight='bold', color='white')
        
        # Highlight RFC
        rfc_idx = models_plot.index('RFC')
        bars[rfc_idx].set_edgecolor('gold')
        bars[rfc_idx].set_linewidth(2.5)
    
    plt.suptitle('BindingDB-Kd Model Comparison (GAN+RFC -> Proposed Model)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_model_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_auc_by_threshold(results_df, models_plot=None):

    if models_plot is None:
        models_plot = ['DTC', 'MLP', 'RFC', 'FCNN', 'MHA-FCNN']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models_plot))
    w = 0.25
    clrs = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (Th_p, clr) in enumerate([(10, clrs[0]), (20, clrs[1]), (30, clrs[2])]):
        sub_p = results_df[results_df['Threshold'] == Th_p]
        vals_p = [
            float(sub_p[sub_p['Model'] == m]['ROC-AUC'].values[0])
            if not sub_p[sub_p['Model'] == m].empty else 0
            for m in models_plot
        ]
        bars = ax.bar(x + (i-1)*w, vals_p, width=w, label=f'Threshold={Th_p}',
                      color=clr, alpha=0.85, edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models_plot, fontsize=11)
    ax.set_ylabel('ROC-AUC (%)', fontsize=11)
    ax.set_ylim(88, 101.5)
    ax.set_title('ROC-AUC Across Models and Thresholds\nBindingDB-Kd Dataset (GAN balanced)',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Threshold (nM)', fontsize=10)
    ax.axhline(99, color='red', linestyle=':', alpha=0.5, label='99% line')
    ax.grid(axis='y', alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_ROC_AUC_by_threshold.png", dpi=150)
    plt.close()


def plot_balancing_effect(bal_df, complexity_df, palette=None):

    if palette is None:
        palette = ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7', '#c4ad66']
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Balancing effect
    no_bal_rfc = bal_df[(bal_df['Balancing'] == 'No Balancing') & (bal_df['Model'] == 'RFC')]
    gan_rfc = bal_df[(bal_df['Balancing'] == 'GAN (Th=10)') & (bal_df['Model'] == 'RFC')]
    
    metrics_cmp = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 
                   'F1-Score', 'ROC-AUC']
    no_vals = [float(no_bal_rfc[m].values[0]) for m in metrics_cmp]
    gan_vals = [float(gan_rfc[m].values[0]) for m in metrics_cmp]
    
    x_b = np.arange(len(metrics_cmp))
    w_b = 0.35
    
    axes[0].bar(x_b - w_b/2, no_vals, w_b, label='No Balancing', 
                color='#d62728', alpha=0.8)
    axes[0].bar(x_b + w_b/2, gan_vals, w_b, label='GAN Balanced', 
                color='#1f77b4', alpha=0.8)
    axes[0].set_xticks(x_b)
    axes[0].set_xticklabels(metrics_cmp, rotation=25, ha='right', fontsize=9)
    axes[0].set_ylim(85, 101)
    axes[0].set_ylabel('Score (%)')
    axes[0].set_title('RFC: No Balancing vs GAN\n(BindingDB-Kd, Th=10)', 
                      fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.25)
    
    # Add improvement annotations
    for i, (nv, gv) in enumerate(zip(no_vals, gan_vals)):
        diff = gv - nv
        axes[0].annotate(f'+{diff:.1f}%', xy=(x_b[i] + w_b/2, gv),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7, color='darkblue', fontweight='bold')
    
    # Complexity chart
    axes[1].barh(complexity_df['Model'], complexity_df['Total_s'],
                 color=palette[:len(complexity_df)], alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Total Execution Time (seconds)', fontsize=10)
    axes[1].set_title('Computational Complexity\n(Training + Prediction Time, Kd)',
                      fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.25)
    
    for i, (model, val) in enumerate(zip(complexity_df['Model'], complexity_df['Total_s'])):
        axes[1].text(val + 0.5, i, f'{val:.1f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_balancing_effect_and_complexity.png", dpi=150)
    plt.close()


# STATISTICAL TESTS

def friedman_test(results_df):
    
    roc_by_model = {}
    for _, row in results_df.iterrows():
        m = row['Model']
        if m not in roc_by_model:
            roc_by_model[m] = []
        roc_by_model[m].append(row['ROC-AUC'])
    
    groups = list(roc_by_model.values())
    stat, pval = friedmanchisquare(*groups)
    
    print(f"\n  Friedman χ²_F statistic : {stat:.4f}")
    print(f"  p-value                  : {pval:.4f}")
    
    return stat, pval


# RESULTS SAVING

def save_all_results(results_df, bal_df, comp_df, cx_df, report_df):

    results_df.to_csv(f"{OUTPUT_DIR}/Kd_Table4_full_results.csv", index=False)
    bal_df.to_csv(f"{OUTPUT_DIR}/Kd_Table9_balancing.csv", index=False)
    comp_df.to_csv(f"{OUTPUT_DIR}/Kd_Table10_ACC_vs_DC.csv", index=False)
    cx_df.to_csv(f"{OUTPUT_DIR}/Kd_Table11_complexity.csv", index=False)
    report_df.to_csv(f"{OUTPUT_DIR}/Kd_Table7_classification.csv", index=False)
    
    print(f"\n  Saved CSV files to ./{OUTPUT_DIR}/:")
    for fname in ['Kd_Table4_full_results.csv', 'Kd_Table9_balancing.csv',
                  'Kd_Table10_ACC_vs_DC.csv', 'Kd_Table11_complexity.csv',
                  'Kd_Table7_classification.csv']:
        print(f"{fname}")