import streamlit as st
import os

def show(model_results, dataset_stats, business_insights):
    if not model_results or not dataset_stats:
        st.error("Data not loaded. Please run the notebook export cells first.")
        return
        
    st.header("Welcome to Flight Delay Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Project Overview
        This dashboard presents results from a comprehensive ML analysis of airline delays using the U.S. Department of Transportation's dataset.
        
        **Objective:** Predict high-delay periods (>25% delay rate) for operational planning
        
        **Key Features:**
        - 📊 **Data Exploration**: Interactive visualizations of flight delay patterns
        - ⚙️ **Preprocessing**: Data cleaning and feature engineering insights
        - 🤖 **Model Training**: Algorithm comparison and selection
        - 📈 **Evaluation**: Performance metrics and business insights
        - 🔮 **Prediction**: Interactive prediction interface
        """)
    
    with col2:
        st.subheader("📊 Dataset")
        st.metric("Records", f"{dataset_stats['total_records']:,}")
        st.metric("Airlines", dataset_stats['airlines'])
        st.metric("Airports", dataset_stats['airports'])
        st.metric("Time Period", dataset_stats['years'])
    
    # Results summary
    st.subheader("🎯 Analysis Results")
    
    if model_results:
        best_model = max(model_results['test_results'].items(), key=lambda x: x[1]['f1_score'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_model[0])
        with col2:
            st.metric("F1-Score", f"{best_model[1]['f1_score']:.4f}")
        with col3:
            st.metric("Accuracy", f"{best_model[1]['accuracy']:.4f}")
        with col4:
            st.metric("ROC-AUC", f"{best_model[1]['roc_auc']:.4f}")
    
    # Key Findings
    if business_insights:
        st.subheader("🔍 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**📈 Model Performance:**")
            for finding in business_insights['key_findings']['model_performance']:
                st.write(f"• {finding}")
                
            st.info("**🕐 Seasonal Patterns:**")
            for pattern in business_insights['key_findings']['seasonal_patterns']:
                st.write(f"• {pattern}")
        
        with col2:
            st.info("**✈️ Operational Insights:**")
            for insight in business_insights['key_findings']['operational_insights']:
                st.write(f"• {insight}")
                
            st.warning("**💼 Business Impact:**")
            st.write(f"• {business_insights['business_impact']['proactive_planning']}")
            st.write(f"• {business_insights['business_impact']['cost_savings']}")
    
    # Navigation guide
    st.subheader("🚀 Navigation Guide")
    st.markdown("""
    **Explore the analysis through these sections:**
    
    1. **📊 Data Exploration** - View dataset insights and delay patterns
    2. **⚙️ Preprocessing** - See data cleaning and preparation steps
    3. **🤖 Model Training** - Compare algorithm performance and selection
    4. **📈 Evaluation** - Detailed performance analysis and business insights
    5. **🔮 Prediction** - Interactive tool to test the model
    
    👈 Use the sidebar to navigate between sections.
    """)
    
    # Model status
    st.subheader("🔧 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_status = "✅ Ready" if os.path.exists('../results/best_model.pkl') else "⚠️ Missing"
        st.metric("Model Status", model_status)
    
    with col2:
        data_status = "✅ Loaded" if dataset_stats else "❌ Missing"
        st.metric("Data Status", data_status)
    
    with col3:
        results_status = "✅ Available" if model_results else "❌ Missing"
        st.metric("Results Status", results_status)