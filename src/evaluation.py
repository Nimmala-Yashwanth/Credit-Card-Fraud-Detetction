"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score,
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.reports = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name="Model"):
        """Comprehensive model evaluation"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'classification_report': report,
            'y_pred_proba': y_pred_proba
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"{model_name} - Performance Metrics")
        print(f"{'='*60}")
        print(f"ROC-AUC Score:        {roc_auc:.4f}")
        print(f"Average Precision:    {avg_precision:.4f}")
        print(f"F1 Score:             {f1:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        print(f"{'='*60}\n")
        
        return results
    
    def plot_confusion_matrix(self, results, save_path='reports/figures/confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = results['confusion_matrix']
        model_name = results['model_name']
        
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=np.array([[f"{val:,}\n({norm:.1%})" 
                           for val, norm in zip(row, norm_row)]
                          for row, norm_row in zip(cm, cm_normalized)]),
            fmt='',
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved confusion matrix: {save_path}")
    
    def plot_roc_curve(self, y_true, results_dict, save_path='reports/figures/roc_curve.png'):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            y_pred_proba = results['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = results['roc_auc']
            
            plt.plot(
                fpr, tpr,
                label=f"{model_name.replace('_', ' ').title()} (AUC = {auc:.4f})",
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved ROC curve: {save_path}")
    
    def plot_precision_recall_curve(self, y_true, results_dict, 
                                     save_path='reports/figures/precision_recall_curve.png'):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            y_pred_proba = results['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = results['avg_precision']
            
            plt.plot(
                recall, precision,
                label=f"{model_name.replace('_', ' ').title()} (AP = {avg_precision:.4f})",
                linewidth=2
            )
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved Precision-Recall curve: {save_path}")
    
    def plot_feature_importance(self, model, feature_names, top_n=20,
                                save_path='reports/figures/feature_importance.png'):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("   ⚠️  Model doesn't have feature importances")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, max(8, top_n * 0.3)))
        
        sns.barplot(
            data=importance_df,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved feature importance: {save_path}")
        
        return importance_df
    
    def plot_comparison(self, results_dict, save_path='reports/figures/model_comparison.png'):
        """Plot comprehensive model comparison"""
        # Prepare data
        metrics = ['roc_auc', 'precision', 'recall', 'f1', 'avg_precision']
        metric_names = ['ROC-AUC', 'Precision', 'Recall', 'F1 Score', 'Avg Precision']
        
        comparison_data = []
        for model_name, results in results_dict.items():
            for metric in metrics:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Metric': metric_names[metrics.index(metric)],
                    'Score': results[metric]
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.barplot(
            data=df,
            x='Metric',
            y='Score',
            hue='Model',
            palette='Set2',
            ax=ax
        )
        
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(title='Model', fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved model comparison: {save_path}")
    
    def generate_report(self, results_dict, output_path='reports/model_report.txt'):
        """Generate comprehensive text report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            for model_name, results in results_dict.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"{model_name.replace('_', ' ').upper()}\n")
                f.write(f"{'-'*80}\n\n")
                
                f.write(f"ROC-AUC Score:        {results['roc_auc']:.4f}\n")
                f.write(f"Average Precision:    {results['avg_precision']:.4f}\n")
                f.write(f"F1 Score:             {results['f1']:.4f}\n")
                f.write(f"Precision:            {results['precision']:.4f}\n")
                f.write(f"Recall:               {results['recall']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"  True Negatives:  {results['true_negatives']:,}\n")
                f.write(f"  False Positives: {results['false_positives']:,}\n")
                f.write(f"  False Negatives: {results['false_negatives']:,}\n")
                f.write(f"  True Positives:  {results['true_positives']:,}\n\n")
                
                f.write("Classification Report:\n")
                report = results['classification_report']
                for label in ['0', '1']:
                    if label in report:
                        f.write(f"  Class {label}:\n")
                        f.write(f"    Precision: {report[label]['precision']:.4f}\n")
                        f.write(f"    Recall:    {report[label]['recall']:.4f}\n")
                        f.write(f"    F1-Score:  {report[label]['f1-score']:.4f}\n")
                        f.write(f"    Support:   {report[label]['support']:,}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"   ✅ Saved evaluation report: {output_path}")
