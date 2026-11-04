"""
Credit Card Fraud Detection - Streamlit Web App
Interactive interface for fraud detection predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.05);
    }
    h1 {
        color: #00d4ff;
        font-weight: 700;
    }
    h2, h3 {
        color: #ffffff;
    }
    .fraud-box {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        return None, None, None
    
    # Find latest models
    rf_files = [f for f in os.listdir(models_dir) if f.startswith('random_forest_')]
    xgb_files = [f for f in os.listdir(models_dir) if f.startswith('xgboost_')]
    feature_files = [f for f in os.listdir(models_dir) if f.startswith('feature_names_')]
    
    if not rf_files or not xgb_files or not feature_files:
        return None, None, None
    
    # Load latest models
    rf_model = joblib.load(os.path.join(models_dir, sorted(rf_files)[-1]))
    xgb_model = joblib.load(os.path.join(models_dir, sorted(xgb_files)[-1]))
    feature_names = joblib.load(os.path.join(models_dir, sorted(feature_files)[-1]))
    
    return rf_model, xgb_model, feature_names

def create_sample_transaction():
    """Create a sample transaction for testing"""
    np.random.seed(None)
    
    # Random transaction
    data = {}
    
    # V features (28 features)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn()
    
    # Amount
    data['Amount'] = np.random.exponential(50)
    
    # Time
    data['Time'] = np.random.uniform(0, 172800)
    
    return data

def engineer_features(data):
    """Apply feature engineering to input data"""
    df = pd.DataFrame([data])
    
    # Amount-based features
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_bin'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        df['Is_small_transaction'] = (df['Amount'] < 10).astype(int)
        df['Is_large_transaction'] = (df['Amount'] > 500).astype(int)
        df['Amount_squared'] = df['Amount'] ** 2
    
    # Time-based features
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        df['Is_business_hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        df['Is_weekend_hours'] = ((df['Hour'] >= 18) & (df['Hour'] <= 22)).astype(int)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # V features statistics
    v_cols = [col for col in df.columns if col.startswith('V')]
    if v_cols:
        df['V_mean'] = df[v_cols].mean(axis=1)
        df['V_std'] = df[v_cols].std(axis=1)
        df['V_max'] = df[v_cols].max(axis=1)
        df['V_min'] = df[v_cols].min(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        df['V_skew'] = df[v_cols].skew(axis=1)
        df['V_kurtosis'] = df[v_cols].kurtosis(axis=1)
        df['V_negative_pct'] = (df[v_cols] < 0).sum(axis=1) / len(v_cols)
        df['V_abs_sum'] = df[v_cols].abs().sum(axis=1)
    
    # Interaction features
    if 'Amount' in df.columns and 'V1' in df.columns:
        df['Amount_V1_interaction'] = df['Amount'] * df['V1']
    if 'Amount' in df.columns and 'V_mean' in df.columns:
        df['Amount_Vmean_interaction'] = df['Amount'] * df['V_mean']
    
    return df

def main():
    """Main app function"""
    
    # Header
    st.markdown("""
        <h1 style='text-align: center;'>💳 Credit Card Fraud Detection System</h1>
        <p style='text-align: center; color: #888; font-size: 18px;'>
            AI-Powered Fraud Detection using Random Forest & XGBoost
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    rf_model, xgb_model, feature_names = load_models()
    
    if rf_model is None:
        st.error("⚠️ Models not found! Please train models first by running: `python train.py`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "Both (Ensemble)"],
            index=0
        )
        
        threshold = st.slider(
            "Fraud Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for classifying as fraud"
        )
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.info(f"**Loaded Models:**\n- Random Forest\n- XGBoost\n\n**Features:** {len(feature_names)}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "📊 Model Info"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Transaction Details")
            
            # Input method
            input_method = st.radio(
                "Input Method",
                ["Manual Entry", "Random Sample", "Upload CSV"],
                horizontal=True
            )
            
            if input_method == "Random Sample":
                if st.button("🎲 Generate Random Transaction", type="primary"):
                    st.session_state.transaction_data = create_sample_transaction()
                
                if 'transaction_data' not in st.session_state:
                    st.session_state.transaction_data = create_sample_transaction()
                
                transaction_data = st.session_state.transaction_data
                
            elif input_method == "Manual Entry":
                st.write("Enter transaction details:")
                
                transaction_data = {}
                
                # Amount and Time
                col_a, col_b = st.columns(2)
                with col_a:
                    transaction_data['Amount'] = st.number_input(
                        "Amount ($)",
                        min_value=0.0,
                        value=100.0,
                        step=10.0
                    )
                with col_b:
                    transaction_data['Time'] = st.number_input(
                        "Time (seconds from start)",
                        min_value=0.0,
                        value=0.0,
                        step=3600.0
                    )
                
                # V features (simplified - show first 5, rest auto-filled)
                st.write("V Features (anonymized transaction features):")
                st.info("For simplicity, enter first 5 V features. Rest will be auto-generated.")
                
                v_cols = st.columns(5)
                for i, col in enumerate(v_cols, 1):
                    with col:
                        transaction_data[f'V{i}'] = st.number_input(
                            f"V{i}",
                            value=0.0,
                            format="%.4f",
                            key=f"v{i}"
                        )
                
                # Auto-fill remaining V features
                for i in range(6, 29):
                    transaction_data[f'V{i}'] = np.random.randn()
                
            else:  # Upload CSV
                uploaded_file = st.file_uploader("Upload transaction CSV", type=['csv'])
                if uploaded_file:
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        if len(df_upload) > 0:
                            transaction_data = df_upload.iloc[0].to_dict()
                            st.success(f"✅ Loaded transaction from CSV")
                        else:
                            st.error("CSV file is empty")
                            transaction_data = create_sample_transaction()
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
                        transaction_data = create_sample_transaction()
                else:
                    transaction_data = create_sample_transaction()
            
            # Display transaction data
            with st.expander("📄 View Raw Transaction Data"):
                df_display = pd.DataFrame([transaction_data])
                st.dataframe(df_display, use_container_width=True)
        
        with col2:
            st.subheader("Key Metrics")
            st.metric("Amount", f"${transaction_data.get('Amount', 0):.2f}")
            
            time_hours = transaction_data.get('Time', 0) / 3600
            st.metric("Time", f"{time_hours:.1f} hours")
            
            hour_of_day = time_hours % 24
            st.metric("Hour of Day", f"{int(hour_of_day)}:00")
            
            # Time category
            if 22 <= hour_of_day or hour_of_day <= 6:
                time_cat = "🌙 Night"
            elif 9 <= hour_of_day <= 17:
                time_cat = "🏢 Business Hours"
            else:
                time_cat = "🌆 Evening"
            
            st.info(time_cat)
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Analyze Transaction", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                # Engineer features
                df_features = engineer_features(transaction_data)
                
                # Ensure all required features are present
                missing_features = set(feature_names) - set(df_features.columns)
                for feature in missing_features:
                    df_features[feature] = 0
                
                # Reorder columns to match training
                df_features = df_features[feature_names]
                
                # Make predictions
                if model_choice == "Random Forest":
                    pred_proba = rf_model.predict_proba(df_features)[0][1]
                    prediction = int(pred_proba >= threshold)
                    
                elif model_choice == "XGBoost":
                    pred_proba = xgb_model.predict_proba(df_features)[0][1]
                    prediction = int(pred_proba >= threshold)
                    
                else:  # Ensemble
                    rf_proba = rf_model.predict_proba(df_features)[0][1]
                    xgb_proba = xgb_model.predict_proba(df_features)[0][1]
                    pred_proba = (rf_proba + xgb_proba) / 2
                    prediction = int(pred_proba >= threshold)
                
                # Display results
                st.markdown("### 🎯 Prediction Result")
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    if prediction == 1:
                        st.markdown("""
                            <div class='fraud-box'>
                                ⚠️ FRAUDULENT TRANSACTION<br>
                                <span style='font-size: 16px;'>Action Required</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='normal-box'>
                                ✅ LEGITIMATE TRANSACTION<br>
                                <span style='font-size: 16px;'>Approved</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col_r2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_proba * 100,
                        title={'text': "Fraud Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("### 📊 Detailed Analysis")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric(
                        "Fraud Probability",
                        f"{pred_proba*100:.2f}%",
                        delta=f"{(pred_proba-threshold)*100:.2f}% vs threshold"
                    )
                
                with col_m2:
                    confidence = abs(pred_proba - 0.5) * 2
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                with col_m3:
                    risk_level = "HIGH" if pred_proba > 0.7 else "MEDIUM" if pred_proba > 0.3 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                with col_m4:
                    st.metric("Model Used", model_choice)
                
                # Recommendation
                st.markdown("### 💡 Recommendation")
                
                if prediction == 1:
                    st.error(f"""
                    **⚠️ HIGH RISK TRANSACTION DETECTED**
                    
                    This transaction has been flagged as potentially fraudulent with {pred_proba*100:.1f}% probability.
                    
                    **Recommended Actions:**
                    - 🔒 Block transaction immediately
                    - 📞 Contact cardholder for verification
                    - 🔍 Investigate transaction details
                    - 📝 Flag account for additional monitoring
                    """)
                else:
                    st.success(f"""
                    **✅ LEGITIMATE TRANSACTION**
                    
                    This transaction appears to be legitimate with {(1-pred_proba)*100:.1f}% confidence.
                    
                    **Recommended Actions:**
                    - ✓ Approve transaction
                    - 📊 Continue normal monitoring
                    """)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Transaction Analysis")
        
        uploaded_batch = st.file_uploader(
            "Upload CSV file with transactions",
            type=['csv'],
            key="batch_upload"
        )
        
        if uploaded_batch:
            try:
                df_batch = pd.read_csv(uploaded_batch)
                st.success(f"✅ Loaded {len(df_batch)} transactions")
                
                # Display sample
                with st.expander("📄 View Sample Data"):
                    st.dataframe(df_batch.head(10))
                
                if st.button("🔮 Analyze All Transactions", type="primary"):
                    with st.spinner(f"Analyzing {len(df_batch)} transactions..."):
                        # Engineer features for all
                        df_features_batch = df_batch.copy()
                        
                        # Apply feature engineering row by row
                        results = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df_batch.iterrows():
                            # Engineer features
                            df_feat = engineer_features(row.to_dict())
                            
                            # Ensure features match
                            for feature in feature_names:
                                if feature not in df_feat.columns:
                                    df_feat[feature] = 0
                            
                            df_feat = df_feat[feature_names]
                            
                            # Predict
                            if model_choice == "Random Forest":
                                prob = rf_model.predict_proba(df_feat)[0][1]
                            elif model_choice == "XGBoost":
                                prob = xgb_model.predict_proba(df_feat)[0][1]
                            else:
                                rf_prob = rf_model.predict_proba(df_feat)[0][1]
                                xgb_prob = xgb_model.predict_proba(df_feat)[0][1]
                                prob = (rf_prob + xgb_prob) / 2
                            
                            pred = int(prob >= threshold)
                            
                            results.append({
                                'Transaction_ID': idx + 1,
                                'Fraud_Probability': prob,
                                'Prediction': 'Fraud' if pred == 1 else 'Normal',
                                'Risk_Level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW'
                            })
                            
                            progress_bar.progress((idx + 1) / len(df_batch))
                        
                        # Create results DataFrame
                        df_results = pd.DataFrame(results)
                        
                        # Display summary
                        st.markdown("### 📊 Batch Analysis Summary")
                        
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        
                        with col_s1:
                            st.metric("Total Transactions", len(df_results))
                        
                        with col_s2:
                            fraud_count = (df_results['Prediction'] == 'Fraud').sum()
                            st.metric("Fraudulent", fraud_count, 
                                     delta=f"{fraud_count/len(df_results)*100:.1f}%")
                        
                        with col_s3:
                            high_risk = (df_results['Risk_Level'] == 'HIGH').sum()
                            st.metric("High Risk", high_risk)
                        
                        with col_s4:
                            avg_prob = df_results['Fraud_Probability'].mean()
                            st.metric("Avg Fraud Prob", f"{avg_prob*100:.1f}%")
                        
                        # Display results
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            data=csv,
                            file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("### 📊 Distribution Analysis")
                        
                        col_v1, col_v2 = st.columns(2)
                        
                        with col_v1:
                            # Prediction distribution
                            fig_pred = px.pie(
                                df_results,
                                names='Prediction',
                                title='Prediction Distribution',
                                color_discrete_map={'Normal': '#00d4ff', 'Fraud': '#ff4444'}
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with col_v2:
                            # Risk level distribution
                            fig_risk = px.pie(
                                df_results,
                                names='Risk_Level',
                                title='Risk Level Distribution',
                                color_discrete_map={'LOW': '#00d4ff', 'MEDIUM': '#ffaa00', 'HIGH': '#ff4444'}
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Probability distribution
                        fig_hist = px.histogram(
                            df_results,
                            x='Fraud_Probability',
                            nbins=50,
                            title='Fraud Probability Distribution',
                            labels={'Fraud_Probability': 'Fraud Probability'},
                            color_discrete_sequence=['#00d4ff']
                        )
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                                          annotation_text=f"Threshold: {threshold}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing batch file: {e}")
    
    # Tab 3: Model Info
    with tab3:
        st.header("Model Information")
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.subheader("🌲 Random Forest")
            st.info("""
            **Algorithm:** Ensemble of Decision Trees
            
            **Key Features:**
            - Handles imbalanced data well
            - Provides feature importance
            - Robust to outliers
            - No feature scaling needed
            
            **Hyperparameters:**
            - n_estimators: 100
            - max_depth: 15
            - min_samples_split: 10
            - class_weight: balanced
            """)
        
        with col_i2:
            st.subheader("🚀 XGBoost")
            st.info("""
            **Algorithm:** Gradient Boosting
            
            **Key Features:**
            - High performance
            - Handles missing values
            - Built-in regularization
            - Scale_pos_weight for imbalance
            
            **Hyperparameters:**
            - n_estimators: 100
            - max_depth: 6
            - learning_rate: 0.1
            - subsample: 0.8
            """)
        
        st.markdown("---")
        
        st.subheader("📊 Feature Engineering")
        st.write(f"**Total Features:** {len(feature_names)}")
        
        with st.expander("View All Features"):
            st.write(feature_names)
        
        st.markdown("""
        **Engineered Features Include:**
        - Amount-based features (log, bins, thresholds)
        - Time-based features (hour, time of day, cyclical encoding)
        - Statistical features from V components
        - Interaction features
        """)
        
        st.markdown("---")
        
        st.subheader("⚖️ Imbalanced Data Handling")
        st.write("""
        The models were trained using **SMOTE (Synthetic Minority Over-sampling Technique)** 
        to handle the highly imbalanced nature of fraud detection data.
        
        **Original Distribution:** ~0.17% fraud
        **After SMOTE:** Balanced 50-50 distribution for training
        """)

if __name__ == "__main__":
    main()
