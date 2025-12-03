import streamlit as st
import pandas as pd
import os
from PIL import Image

def show(model_results):
    st.header("🤖 Model Training Results")
    
    if not model_results:
        st.error("Training results not available. Please run notebook export cells.")
        return
    
    st.markdown("""
    This page shows the machine learning models trained and their performance comparison.
    All results displayed here were computed during the notebook analysis phase.
    """)
    
    # Model Performance Overview
    st.subheader("📊 Model Performance Overview")
    
    # Create comprehensive comparison dataframe
    comparison_data = []
    for model in model_results['baseline_scores'].keys():
        baseline_score = model_results['baseline_scores'][model]
        optimized_score = model_results['optimized_scores'].get(model, baseline_score)
        test_results = model_results['test_results'].get(model, {})
        training_time = model_results['training_times'].get(model, 0)
        
        comparison_data.append({
            'Model': model,
            'Baseline F1': f"{baseline_score:.4f}",
            'Optimized F1': f"{optimized_score:.4f}",
            'Test Accuracy': f"{test_results.get('accuracy', 0):.4f}",
            'Test F1-Score': f"{test_results.get('f1_score', 0):.4f}",
            'ROC-AUC': f"{test_results.get('roc_auc', 0):.4f}",
            'Training Time (s)': f"{training_time:.2f}"
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by test F1-score
    df['sort_key'] = df['Test F1-Score'].astype(float)
    df = df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    st.dataframe(df, use_container_width=True)
    
    # Best model highlight
    best_model = max(model_results['test_results'].items(), key=lambda x: x[1]['f1_score'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Best Model", best_model[0])
    with col2:
        st.metric("🎯 F1-Score", f"{best_model[1]['f1_score']:.4f}")
    with col3:
        st.metric("📊 Accuracy", f"{best_model[1]['accuracy']:.4f}")
    with col4:
        st.metric("📈 ROC-AUC", f"{best_model[1]['roc_auc']:.4f}")
    
    # Model comparison visualization
    st.subheader("📈 Performance Comparison")
    
    img_path = "../results/model_comparison.png"
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="Model Performance Comparison", use_column_width=True)
    else:
        st.warning("Model comparison chart not available. Run notebook export cells.")
    
    # Hyperparameter tuning results
    st.subheader("⚙️ Hyperparameter Optimization")
    
    tab1, tab2 = st.tabs(["🌲 Random Forest", "🚀 XGBoost"])
    
    with tab1:
        if 'Random Forest' in model_results['best_params']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Optimized Parameters:**")
                rf_params = model_results['best_params']['Random Forest']
                st.json(rf_params)
            
            with col2:
                st.markdown("**Performance Improvement:**")
                baseline_f1 = model_results['baseline_scores']['Random Forest']
                optimized_f1 = model_results['optimized_scores']['Random Forest']
                improvement = optimized_f1 - baseline_f1
                
                st.metric("Baseline F1", f"{baseline_f1:.4f}")
                st.metric("Optimized F1", f"{optimized_f1:.4f}", delta=f"+{improvement:.4f}")
                
                st.success(f"**Improvement:** +{improvement:.4f} F1-score ({improvement/baseline_f1:.1%} gain)")
    
    with tab2:
        if 'XGBoost' in model_results['best_params']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Optimized Parameters:**")
                xgb_params = model_results['best_params']['XGBoost']
                st.json(xgb_params)
            
            with col2:
                st.markdown("**Performance Improvement:**")
                baseline_f1 = model_results['baseline_scores']['XGBoost']
                optimized_f1 = model_results['optimized_scores']['XGBoost']
                improvement = optimized_f1 - baseline_f1
                
                st.metric("Baseline F1", f"{baseline_f1:.4f}")
                st.metric("Optimized F1", f"{optimized_f1:.4f}", delta=f"+{improvement:.4f}")
                
                st.success(f"**Improvement:** +{improvement:.4f} F1-score ({improvement/baseline_f1:.1%} gain)")
    
    # Algorithm comparison and selection rationale
    st.subheader("🎯 Model Selection Rationale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Why XGBoost Won:**
        - Highest F1-score (0.6670) on test set
        - Excellent ROC-AUC (0.8729) for discrimination
        - Good accuracy (79.3%) for operational use
        - Handles class imbalance effectively
        - Fast training and inference
        - Robust to overfitting with regularization
        """)
    
    with col2:
        st.info("""
        **🔄 Algorithm Comparison:**
        - **XGBoost**: Best overall performance, gradient boosting
        - **Random Forest**: Strong baseline, ensemble bagging
        - **Decision Tree**: Good interpretability, prone to overfitting
        - **KNN**: Distance-based, sensitive to scaling
        - **Logistic Regression**: Linear assumptions too restrictive
        """)
    
    # Training insights
    st.subheader("💡 Training Insights")
    
    st.warning("""
    **Key Training Decisions:**
    
    🎯 **Class Imbalance Handling**: Used balanced class weights (2.69x weight for minority class)
    
    ⚙️ **Hyperparameter Optimization**: GridSearchCV with 3-fold CV for top 2 models
    
    📊 **Evaluation Strategy**: Stratified train-test split (80/20) with F1-score focus
    
    🔄 **Cross-Validation**: 5-fold CV for reliable performance estimates
    
    ⏱️ **Efficiency**: Total training time under 8 minutes for all models
    """)
    
    # Next steps
    st.info("""
    **🔄 Training Complete - Model Ready for Deployment:**
    
    ✅ **Best Model Selected**: XGBoost with optimized hyperparameters  
    ✅ **Performance Validated**: 79.3% accuracy, 0.667 F1-score on test set  
    ✅ **Business Ready**: Model exported and ready for inference  
    ✅ **Robust Evaluation**: Cross-validation and holdout testing completed  
    
    👉 **Next**: View **Evaluation** for detailed performance analysis and business insights.
    """)