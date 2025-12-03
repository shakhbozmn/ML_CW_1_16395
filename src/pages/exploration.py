import streamlit as st
import os
from PIL import Image

def show(dataset_stats):
    st.header("📊 Data Exploration")
    
    if not dataset_stats:
        st.error("Dataset stats not available.")
        return
    
    st.markdown("""
    This page presents key findings from the exploratory data analysis conducted in the Jupyter notebook.
    All visualizations and insights shown here were discovered during the research phase.
    """)
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{dataset_stats['total_records']:,}")
    with col2:
        st.metric("Features", dataset_stats['features'])
    with col3:
        st.metric("Airlines", dataset_stats['airlines'])
    with col4:
        st.metric("Airports", dataset_stats['airports'])
    
    # Missing values info
    st.subheader("📄 Data Quality Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Data Completeness:**
        - Time Period: {dataset_stats['years']}
        - Missing values handled: ✅
        - Data quality checks: ✅
        - Outlier treatment: ✅
        """)
    
    with col2:
        class_dist = dataset_stats['class_distribution']
        st.success(f"""
        **Target Distribution:**
        - Normal Operations: {class_dist['0']:.1%}
        - High Delay Periods: {class_dist['1']:.1%}
        - Balance Strategy: Class weighting applied
        """)
    
    # Visualizations from exported files
    st.subheader("📊 Key Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Delay Patterns", "✈️ Carrier Analysis", "📅 Monthly Trends", "🔗 Data Insights"])
    
    with tab1:
        st.markdown("### Arrival Delay Distribution")
        
        # Load delay distribution image
        img_path = "../results/delay_distribution.png"
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption="Distribution of Arrival Delays", use_column_width=True)
        else:
            st.warning("Delay distribution visualization not available. Run notebook export cells.")
        
        st.info("""
        **Key Insights:**
        - Most flights have minimal delays (right-skewed distribution)
        - Long tail of extreme delays exists
        - Majority of delays are under 60 minutes
        - Clear patterns for operational planning
        """)
    
    with tab2:
        st.markdown("### Carrier Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Flight Volume by Carrier**")
            img_path = "../results/top_carriers.png"
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption="Top 10 Carriers by Volume", use_column_width=True)
        
        with col2:
            st.markdown("**Average Delay by Carrier**")
            img_path = "../results/carrier_delays.png"
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption="Carrier Delay Performance", use_column_width=True)
        
        st.success("""
        **Carrier Insights:**
        - Significant variation in delay performance between carriers
        - Volume doesn't always correlate with delay rates
        - Hub-and-spoke vs point-to-point models show different patterns
        - Regional vs major carriers have distinct characteristics
        """)
    
    with tab3:
        st.markdown("### Seasonal and Monthly Patterns")
        
        img_path = "../results/average_delay_by_month.png"
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption="Average Delay by Month", use_column_width=True)
        else:
            st.warning("Monthly trends visualization not available.")
        
        st.warning("""
        **Seasonal Insights:**
        - **Summer Peak**: June-August show highest delays (weather + traffic)
        - **Winter Impact**: December-January delays due to weather conditions  
        - **Spring Minimum**: March-May typically have lowest delay rates
        - **Holiday Effect**: November and December show increased disruptions
        """)
    
    with tab4:
        st.markdown("### Data Quality and Relationships")
        
        st.info("""
        **Data Quality Findings:**
        - 409K+ flight records spanning 2003-2025
        - Missing values concentrated in delay-related columns
        - No duplicate records found
        - Strong seasonal and operational patterns identified
        """)
        
        st.success("""
        **Feature Relationships Discovered:**
        - Strong correlations between different delay types
        - Flight volume correlates with total delays
        - Cancellations and diversions show positive correlation
        - Weather delays peak during specific months
        """)
        
        st.warning("""
        **Challenges Identified:**
        - Class imbalance in delay categories (addressed with weighting)
        - Missing values in delay cause columns (handled with imputation)
        - Outliers in delay minutes (treated with IQR method)
        - Categorical variables required encoding
        """)
    
    # Summary
    st.subheader("📊 EDA Summary")
    st.info("""
    **🔄 These findings informed our modeling approach:**
    - Missing value imputation strategy based on business logic
    - Outlier treatment using IQR method to preserve distribution shape
    - Feature engineering for temporal and operational patterns
    - Target variable definition using 25% delay rate threshold
    - Model selection criteria focusing on business impact
    
    👉 **Next**: View the **Preprocessing** page to see how these insights guided data preparation.
    """)