"""
Data Preprocessing Module
Handle data cleaning, scaling, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # Better for outliers
        self.imputer = SimpleImputer(strategy='median')
        self.fitted = False
        
    def clean_data(self, df):
        """Clean and validate data"""
        print("🧹 Cleaning data...")
        
        # Make a copy
        df = df.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"   ℹ️  Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"   ℹ️  Found {missing.sum()} missing values")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"   ✅ Data cleaned: {len(df)} rows remaining")
        return df
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using RobustScaler"""
        print("📏 Scaling features...")
        
        # Fit on training data
        if not self.fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def detect_outliers(self, df, threshold=3):
        """Detect outliers using Z-score"""
        from scipy import stats
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        outliers = (z_scores > threshold).sum(axis=1)
        
        print(f"📊 Outlier detection (Z-score > {threshold}):")
        print(f"   Found {(outliers > 0).sum()} rows with outliers")
        
        return outliers > 0
    
    def split_features_target(self, df, target_col='Class'):
        """Split features and target variable"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        return X, y
