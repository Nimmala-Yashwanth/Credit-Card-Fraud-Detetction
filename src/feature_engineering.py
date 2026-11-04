"""
Feature Engineering Module
Create advanced features for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Create and engineer features for better fraud detection"""
    
    def __init__(self):
        self.pca = None
        
    def create_features(self, df):
        """Create advanced features"""
        print("🔧 Engineering features...")
        
        df = df.copy()
        
        # Amount-based features
        if 'Amount' in df.columns:
            df = self._create_amount_features(df)
        
        # Time-based features (if Time column exists)
        if 'Time' in df.columns:
            df = self._create_time_features(df)
        
        # Statistical features from V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            df = self._create_statistical_features(df, v_cols)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        print(f"   ✅ Features created: {len(df.columns)} total features")
        return df
    
    def _create_amount_features(self, df):
        """Create amount-based features"""
        # Log transformation for amount (handle zeros)
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Amount bins
        df['Amount_bin'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        
        # Is small/large transaction
        df['Is_small_transaction'] = (df['Amount'] < 10).astype(int)
        df['Is_large_transaction'] = (df['Amount'] > 500).astype(int)
        
        # Amount squared (for non-linear relationships)
        df['Amount_squared'] = df['Amount'] ** 2
        
        return df
    
    def _create_time_features(self, df):
        """Create time-based features"""
        # Convert time to hours
        df['Hour'] = (df['Time'] / 3600) % 24
        
        # Time of day features
        df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        df['Is_business_hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        df['Is_weekend_hours'] = ((df['Hour'] >= 18) & (df['Hour'] <= 22)).astype(int)
        
        # Cyclical encoding for hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        return df
    
    def _create_statistical_features(self, df, v_cols):
        """Create statistical features from V columns"""
        # Mean and std of V features
        df['V_mean'] = df[v_cols].mean(axis=1)
        df['V_std'] = df[v_cols].std(axis=1)
        df['V_max'] = df[v_cols].max(axis=1)
        df['V_min'] = df[v_cols].min(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        
        # Skewness and kurtosis
        df['V_skew'] = df[v_cols].skew(axis=1)
        df['V_kurtosis'] = df[v_cols].kurtosis(axis=1)
        
        # Percentage of negative values
        df['V_negative_pct'] = (df[v_cols] < 0).sum(axis=1) / len(v_cols)
        
        # Sum of absolute values
        df['V_abs_sum'] = df[v_cols].abs().sum(axis=1)
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features"""
        # If we have both Amount and V features
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        if 'Amount' in df.columns and len(v_cols) > 0:
            # Amount * V1 (often important)
            if 'V1' in df.columns:
                df['Amount_V1_interaction'] = df['Amount'] * df['V1']
            
            # Amount * mean(V features)
            if 'V_mean' in df.columns:
                df['Amount_Vmean_interaction'] = df['Amount'] * df['V_mean']
        
        return df
    
    def apply_pca(self, X, n_components=10):
        """Apply PCA for dimensionality reduction"""
        print(f"🔄 Applying PCA (n_components={n_components})...")
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        
        # Create DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        X_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
        
        print(f"   ✅ Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        return X_pca
    
    def select_features(self, X, y, method='importance', n_features=20):
        """Select top features based on various methods"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        print(f"🎯 Selecting top {n_features} features using {method}...")
        
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'importance':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # Get feature importances
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importances.head(n_features)['feature'].tolist()
            X_selected = X[selected_features]
        
        print(f"   ✅ Selected features: {selected_features[:5]}...")
        
        return X_selected, selected_features
