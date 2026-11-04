"""
Credit Card Fraud Detection - Main Training Script
Train Random Forest and XGBoost models with imbalanced data handling
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator

class FraudDetectionTrainer:
    """Main class for training fraud detection models"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self, sample_size=None):
        """Load and prepare data for training"""
        print("\n" + "="*80)
        print("📊 LOADING AND PREPARING DATA")
        print("="*80 + "\n")
        
        if self.data_path and os.path.exists(self.data_path):
            print(f"📁 Loading data from: {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            print("⚠️  No data file found. Generating synthetic data...")
            df = self._generate_synthetic_data(sample_size or 100000)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Total transactions: {df.shape[0]:,}")
        print(f"   Features: {df.shape[1]}")
        print(f"   Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
        print(f"   Normal cases: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")
        
        # Clean data
        df = self.preprocessor.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineer.create_features(df)
        
        # Split features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training set: {len(self.X_train):,} samples")
        print(f"   Test set: {len(self.X_test):,} samples")
        print(f"   Train fraud rate: {self.y_train.mean()*100:.3f}%")
        print(f"   Test fraud rate: {self.y_test.mean()*100:.3f}%")
        
        return self
        
    def _generate_synthetic_data(self, n_samples=100000):
        """Generate synthetic credit card transaction data"""
        print(f"\n🔄 Generating {n_samples:,} synthetic transactions...")
        
        np.random.seed(42)
        
        # Realistic fraud rate: 0.17%
        fraud_rate = 0.0017
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        print(f"   Normal transactions: {n_normal:,}")
        print(f"   Fraudulent transactions: {n_fraud:,}")
        
        # Generate V features (anonymized PCA components)
        # Normal transactions: standard normal distribution
        normal_v = np.random.randn(n_normal, 28)
        
        # Fraudulent transactions: shifted and scaled distributions
        fraud_v = np.random.randn(n_fraud, 28)
        # Make fraud patterns distinct
        fraud_shift = np.random.uniform(-3, 3, 28)
        fraud_scale = np.random.uniform(0.5, 2.5, 28)
        fraud_v = fraud_v * fraud_scale + fraud_shift
        
        # Combine
        V_data = np.vstack([normal_v, fraud_v])
        
        # Generate Amount (log-normal distribution)
        normal_amounts = np.random.lognormal(3.5, 1.2, n_normal)
        # Fraudulent amounts tend to be larger
        fraud_amounts = np.random.lognormal(4.5, 1.5, n_fraud)
        amounts = np.hstack([normal_amounts, fraud_amounts])
        
        # Generate Time (seconds from start)
        times = np.random.uniform(0, 172800, n_samples)  # 48 hours
        
        # Create labels
        labels = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Create DataFrame
        columns = [f'V{i}' for i in range(1, 29)]
        df = pd.DataFrame(V_data, columns=columns)
        df['Time'] = times
        df['Amount'] = amounts
        df['Class'] = labels.astype(int)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ Synthetic data generated successfully!")
        
        return df
    
    def handle_imbalance(self, method='smote'):
        """Handle class imbalance using various techniques"""
        print("\n" + "="*80)
        print(f"⚖️  HANDLING CLASS IMBALANCE - {method.upper()}")
        print("="*80 + "\n")
        
        original_counts = self.y_train.value_counts()
        print(f"📊 Original Distribution:")
        print(f"   Normal (0): {original_counts[0]:,} ({original_counts[0]/len(self.y_train)*100:.2f}%)")
        print(f"   Fraud (1):  {original_counts[1]:,} ({original_counts[1]/len(self.y_train)*100:.2f}%)")
        print(f"   Imbalance ratio: 1:{int(original_counts[0]/original_counts[1])}")
        
        if method == 'smote':
            print(f"\n🔄 Applying SMOTE (Synthetic Minority Over-sampling)...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train, self.y_train
            )
            
        elif method == 'smotetomek':
            print(f"\n🔄 Applying SMOTETomek (Hybrid method)...")
            smt = SMOTETomek(random_state=42)
            self.X_train_resampled, self.y_train_resampled = smt.fit_resample(
                self.X_train, self.y_train
            )
            
        elif method == 'none':
            print(f"\n⚠️  No resampling - using class weights instead")
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
            return self
        
        new_counts = pd.Series(self.y_train_resampled).value_counts()
        print(f"\n📊 Resampled Distribution:")
        print(f"   Normal (0): {new_counts[0]:,} ({new_counts[0]/len(self.y_train_resampled)*100:.2f}%)")
        print(f"   Fraud (1):  {new_counts[1]:,} ({new_counts[1]/len(self.y_train_resampled)*100:.2f}%)")
        print(f"   New imbalance ratio: 1:{int(new_counts[0]/new_counts[1]) if new_counts[1] > 0 else 'inf'}")
        print(f"   ✅ Resampling completed!")
        
        return self
    
    def train_random_forest(self, use_resampled=True):
        """Train Random Forest classifier"""
        print("\n" + "="*80)
        print("🌲 TRAINING RANDOM FOREST CLASSIFIER")
        print("="*80 + "\n")
        
        X_train = self.X_train_resampled if use_resampled else self.X_train
        y_train = self.y_train_resampled if use_resampled else self.y_train
        
        print(f"📊 Training Configuration:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Using resampled data: {use_resampled}")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced' if not use_resampled else None,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print(f"\n🔄 Training Random Forest...")
        rf_model.fit(X_train, y_train)
        
        # Predictions
        print(f"🔮 Making predictions on test set...")
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        results = self.evaluator.evaluate_model(
            self.y_test, y_pred, y_pred_proba, "Random Forest"
        )
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = results
        
        # Plot feature importance
        self.evaluator.plot_feature_importance(
            rf_model,
            list(self.X_train.columns),
            top_n=20,
            save_path='reports/figures/rf_feature_importance.png'
        )
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            results,
            save_path='reports/figures/rf_confusion_matrix.png'
        )
        
        return self
    
    def train_xgboost(self, use_resampled=True):
        """Train XGBoost classifier"""
        print("\n" + "="*80)
        print("🚀 TRAINING XGBOOST CLASSIFIER")
        print("="*80 + "\n")
        
        X_train = self.X_train_resampled if use_resampled else self.X_train
        y_train = self.y_train_resampled if use_resampled else self.y_train
        
        print(f"📊 Training Configuration:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Using resampled data: {use_resampled}")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if not use_resampled else 1
        
        print(f"   Scale_pos_weight: {scale_pos_weight:.2f}")
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
        
        print(f"\n🔄 Training XGBoost...")
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        print(f"🔮 Making predictions on test set...")
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        results = self.evaluator.evaluate_model(
            self.y_test, y_pred, y_pred_proba, "XGBoost"
        )
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = results
        
        # Plot feature importance
        self.evaluator.plot_feature_importance(
            xgb_model,
            list(self.X_train.columns),
            top_n=20,
            save_path='reports/figures/xgb_feature_importance.png'
        )
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            results,
            save_path='reports/figures/xgb_confusion_matrix.png'
        )
        
        return self
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON")
        print("="*80 + "\n")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'ROC-AUC': f"{results['roc_auc']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1 Score': f"{results['f1']:.4f}",
                'Avg Precision': f"{results['avg_precision']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Generate comparison visualizations
        print("\n📊 Generating comparison visualizations...")
        self.evaluator.plot_roc_curve(self.y_test, self.results)
        self.evaluator.plot_precision_recall_curve(self.y_test, self.results)
        self.evaluator.plot_comparison(self.results)
        self.evaluator.generate_report(self.results)
        
        return comparison_df
    
    def save_models(self, output_dir='models'):
        """Save trained models and metadata"""
        print("\n" + "="*80)
        print(f"💾 SAVING MODELS")
        print("="*80 + "\n")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            filepath = f"{output_dir}/{model_name}_{timestamp}.joblib"
            joblib.dump(model, filepath)
            print(f"   ✅ Saved: {filepath}")
        
        # Save feature names
        feature_filepath = f"{output_dir}/feature_names_{timestamp}.joblib"
        joblib.dump(list(self.X_train.columns), feature_filepath)
        print(f"   ✅ Saved feature names: {feature_filepath}")
        
        # Save scaler
        scaler_filepath = f"{output_dir}/scaler_{timestamp}.joblib"
        joblib.dump(self.preprocessor.scaler, scaler_filepath)
        print(f"   ✅ Saved scaler: {scaler_filepath}")
        
        print(f"\n   📁 All models saved to: {output_dir}/")
        
        return self

def main():
    """Main execution function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        CREDIT CARD FRAUD DETECTION TRAINING PIPELINE              ║
    ║        Using Random Forest & XGBoost with SMOTE                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(data_path='data/creditcard.csv')
    
    # Load and prepare data
    trainer.load_and_prepare_data(sample_size=100000)
    
    # Handle imbalanced data using SMOTE
    trainer.handle_imbalance(method='smote')
    
    # Train models
    trainer.train_random_forest(use_resampled=True)
    trainer.train_xgboost(use_resampled=True)
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Output Locations:")
    print("   Models: models/")
    print("   Visualizations: reports/figures/")
    print("   Evaluation Report: reports/model_report.txt")
    print("\n🚀 Next: Run 'streamlit run app/app.py' to launch the web interface!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
