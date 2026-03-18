import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from evaluation import compute_metrics
from config import (
    SEED, OUTPUT_DIR, BINARIZATION_THRESHOLDS, PRIMARY_THRESHOLD,
    PAPER_REFERENCES
)

from data_loader import (
    load_bindingdb_kd, perform_eda, binarize_dataset,
    plot_class_distribution, train_test_split_stratified
)

from feature_engineering import (
    compute_maccs_fingerprints, compute_aac_features, compute_dc_features,
    combine_features, standardize_features, plot_correlation_heatmap
)

from gan_balancer import balance_dataset_with_gan, plot_balancing_results

from train import (
    run_experiment_for_threshold, run_complexity_analysis,
    generate_classification_reports
)

from evaluation import (
    plot_confusion_matrices, plot_model_comparison,
    plot_roc_auc_by_threshold, plot_balancing_effect,
    friedman_test, save_all_results
)

from models import train_torch_model, FCNN, MHAFCNN, predict_torch

# MAIN PIPELINE

def main():    
    print("\n" + "═" * 78)
    print("  DRUG-TARGET INTERACTION PREDICTION PIPELINE")
    print("  Reproducing Talukder et al. (Scientific Reports 2025)")
    print("═" * 78)
    
    
    # SECTION 1-2: DATA LOADING & EDA
    data_obj, raw_df = load_bindingdb_kd()
    eda_stats = perform_eda(raw_df)
    
    
    # SECTION 3: DATA PREPROCESSING (Binarization)
    binarized_dfs = binarize_dataset()
    df_main = binarized_dfs[PRIMARY_THRESHOLD].copy()
    
    # Plot class distribution before balancing
    plot_class_distribution(df_main, "BEFORE GAN")
    
    # SECTION 4: FEATURE ENGINEERING
    # Compute MACCS fingerprints
    maccs_arr, valid_indices, df_main = compute_maccs_fingerprints(df_main)
    y = df_main['Y'].values.astype(np.int64)
    
    # Compute AAC features
    aac_arr = compute_aac_features(df_main)
    
    # Compute DC features
    dc_arr = compute_dc_features(df_main)
    
    # Combine features
    feature_sets = combine_features(maccs_arr, aac_arr, dc_arr)
    
    # Standardize ACC features (primary feature set)
    X_acc_scaled, scaler_acc = standardize_features(feature_sets['ACC'])
    
    # Plot correlation heatmap
    plot_correlation_heatmap(X_acc_scaled, title_suffix="BindingDB-Kd Dataset")
    
    # SECTION 5: GAN-BASED DATA BALANCING

    X_balanced, y_balanced = balance_dataset_with_gan(X_acc_scaled, y)
    plot_balancing_results(y, y_balanced)
    
    # SECTION 6: TRAIN/TEST SPLIT

    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split_stratified(
        X_balanced, y_balanced
    )
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_stratified(
        X_acc_scaled, y
    )
    
    # SECTION 8: FULL EXPERIMENT 
    print("\n" + "=" * 65)
    print("FULL EXPERIMENT — TABLE 4 (BindingDB-Kd)")
    print("=" * 65)
    
    all_results = []
    for Th in BINARIZATION_THRESHOLDS:
        results = run_experiment_for_threshold(Th, X_balanced, y_balanced)
        all_results.extend(results)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 65)
    print("TABLE 4: Performance Analysis of ML/DL Models on BindingDB-Kd")
    print("=" * 65)
    
    cols_t4 = ['Threshold', 'Model', 'Accuracy', 'Precision', 'Sensitivity',
               'Specificity', 'F1-Score', 'Kappa', 'MCC', 'ROC-AUC', 'MAE', 'MSE', 'RMSE']
    
    for Th in BINARIZATION_THRESHOLDS:
        sub = results_df[results_df['Threshold'] == Th][cols_t4]
        print(f"\n  Threshold = {Th}:")
        print(sub.to_string(index=False))
    
    # SECTION 9: CONFUSION MATRICES
    print("\n" + "=" * 65)
    print("CONFUSION MATRICES")
    print("=" * 65)
    
    # Retrain models for confusion matrices
    cm_preds_dict = {}
    
    # Sklearn models
    from models import get_sklearn_model
    for name in ['DTC', 'MLP', 'RFC']:
        model = get_sklearn_model(name)
        model.fit(X_train_bal, y_train_bal)
        cm_preds_dict[name] = model.predict(X_test_bal)
    
    # Deep learning models
    for dl_name, ModelCls in [('FCNN', FCNN), ('MHA-FCNN', MHAFCNN)]:
        model = train_torch_model(ModelCls(X_train_bal.shape[1]), 
                                  X_train_bal, y_train_bal, epochs=50)
        preds, _ = predict_torch(model, X_test_bal)
        cm_preds_dict[dl_name] = preds
    
    # Plot confusion matrices
    plot_confusion_matrices(
        y_test_bal, cm_preds_dict,
        model_order=['DTC', 'MLP', 'RFC', 'FCNN', 'MHA-FCNN'],
        title='Fig. 5: Confusion Matrices — ML/DL on BindingDB-Kd (GAN balanced, Th=10)'
    )
    
    # SECTION 10: CLASSIFICATION REPORT 
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT (Table 7)")
    print("=" * 65)
    
    report_df = generate_classification_reports(y_test_bal, cm_preds_dict)
    print("\n  Classification Report (Table 7):")
    print(report_df.to_string(index=False))
    
    # SECTION 11: FRIEDMAN TEST
    print("\n" + "=" * 65)
    print("FRIEDMAN STATISTICAL TEST")
    print("=" * 65)
    
    stat, pval = friedman_test(results_df)
    
    # SECTION 12: EFFECT OF BALANCING 

    print("\n" + "=" * 65)
    print("EFFECT OF DATA BALANCING (Table 9)")
    print("=" * 65)
    
    bal_rows = []
    
    # Without balancing
    for name in ['DTC', 'MLP', 'RFC']:
        model = get_sklearn_model(name)
        model.fit(X_train_raw, y_train_raw)
        preds = model.predict(X_test_raw)
        proba = model.predict_proba(X_test_raw)[:, 1]
        metrics = compute_metrics(y_test_raw, preds, proba)
        bal_rows.append({'Balancing': 'No Balancing', 'Model': name, **metrics})
    
    # With balancing
    for _, row in results_df[results_df['Threshold'] == PRIMARY_THRESHOLD].iterrows():
        if row['Model'] in ['DTC', 'MLP', 'RFC']:
            bal_rows.append({
                'Balancing': f'GAN (Th={PRIMARY_THRESHOLD})',
                'Model': row['Model'],
                'Accuracy': row['Accuracy'],
                'Precision': row['Precision'],
                'Sensitivity': row['Sensitivity'],
                'Specificity': row['Specificity'],
                'F1-Score': row['F1-Score'],
                'ROC-AUC': row['ROC-AUC'],
            })
    
    bal_df = pd.DataFrame(bal_rows)
    cols_bal = ['Balancing','Model','Accuracy','Precision','Sensitivity',
                'Specificity','F1-Score','ROC-AUC']
    print("\n  Table 9 — Effect of Data Balancing:")
    print(bal_df[cols_bal].to_string(index=False))
    
    # SECTION 13: ACC vs DC COMPARISON (Table 10)
    print("\n" + "=" * 65)
    print("ACC vs DC COMPARISON (Table 10)")
    print("=" * 65)
    
    # Prepare DC features with balancing
    X_dc_scaled, _ = standardize_features(feature_sets['DC'])
    X_dc_balanced, y_dc_balanced = balance_dataset_with_gan(X_dc_scaled, y, verbose=False)
    X_tr_dc, X_te_dc, y_tr_dc, y_te_dc = train_test_split_stratified(
        X_dc_balanced, y_dc_balanced
    )
    
    comp_rows = []
    
    # ACC results
    for _, row in results_df[results_df['Threshold'] == PRIMARY_THRESHOLD].iterrows():
        if row['Model'] in ['DTC', 'MLP', 'RFC']:
            comp_rows.append({
                'Composition': 'ACC',
                'Model': row['Model'],
                'Accuracy': row['Accuracy'],
                'Precision': row['Precision'],
                'Sensitivity': row['Sensitivity'],
                'Specificity': row['Specificity'],
                'F1-Score': row['F1-Score'],
                'ROC-AUC': row['ROC-AUC'],
            })
    
    # DC results
    for name in ['DTC', 'MLP', 'RFC']:
        model = get_sklearn_model(name)
        model.fit(X_tr_dc, y_tr_dc)
        preds = model.predict(X_te_dc)
        proba = model.predict_proba(X_te_dc)[:, 1]
        metrics = compute_metrics(y_te_dc, preds, proba)
        comp_rows.append({'Composition': 'DC', 'Model': name, **metrics})
    
    comp_df = pd.DataFrame(comp_rows)
    cols_comp = ['Composition','Model','Accuracy','Precision','Sensitivity',
                 'Specificity','F1-Score','ROC-AUC']
    print("\n  Table 10 — ACC vs DC Comparison:")
    print(comp_df[cols_comp].to_string(index=False))
    
    # SECTION 14: COMPLEXITY ANALYSIS (Table 11)
    cx_df = run_complexity_analysis(X_train_bal, y_train_bal, X_test_bal, y_test_bal)
    print("\n  Table 11 — Complexity Analysis:")
    print(cx_df.to_string(index=False))
    
    # SECTION 15: SUMMARY VISUALIZATIONS
    print("\n" + "=" * 65)
    print("SUMMARY VISUALIZATIONS")
    print("=" * 65)
    
    plot_model_comparison(results_df)
    plot_roc_auc_by_threshold(results_df)
    plot_balancing_effect(bal_df, cx_df)
    
    # SECTION 16: SAVE RESULTS
    save_all_results(results_df, bal_df, comp_df, cx_df, report_df)
    
    # SECTION 17: MASTER SUMMARY
    print("\n" + "═" * 78)
    print("  MASTER SUMMARY — GAN+RFC MODEL (Proposed, BindingDB-Kd)")
    print("═" * 78)
    
    rfc_rows = results_df[results_df['Model'] == 'RFC']
    print(f"\n  {'Th':>4}  {'Acc%':>7}  {'Prec%':>7}  {'Sens%':>7}  {'Spec%':>7}  "
          f"{'F1%':>7}  {'ROC-AUC%':>9}  {'Kappa':>7}  {'MCC':>7}")
    print("  " + "─" * 74)
    for _, row in rfc_rows.iterrows():
        print(f"  {int(row['Threshold']):>4}  {row['Accuracy']:>7.2f}  {row['Precision']:>7.2f}  "
              f"{row['Sensitivity']:>7.2f}  {row['Specificity']:>7.2f}  {row['F1-Score']:>7.2f}  "
              f"{row['ROC-AUC']:>9.2f}  {row['Kappa']:>7.2f}  {row['MCC']:>7.2f}")



if __name__ == "__main__":
    main()