import streamlit as st

def show(dataset_stats, preprocessing_info):
    st.header("⚙️ Data Preprocessing")
    
    if not dataset_stats or not preprocessing_info:
        st.error("Preprocessing information not available. Please run notebook export cells.")
        return
    
    st.markdown("""
    This page demonstrates the preprocessing choices made during the analysis phase. 
    All steps shown here were implemented in the Jupyter notebook to prepare the data for machine learning.
    """)
    
    # Preprocessing Pipeline Overview
    st.subheader("🔄 Preprocessing Pipeline")
    
    pipeline_steps = [
        "1️⃣ **Missing Value Analysis & Handling**",
        "2️⃣ **Data Quality Checks & Corrections**", 
        "3️⃣ **Outlier Detection & Treatment**",
        "4️⃣ **Feature Engineering & Creation**",
        "5️⃣ **Target Variable Definition**",
        "6️⃣ **Categorical Encoding**",
        "7️⃣ **Feature Scaling & Normalization**",
        "8️⃣ **Data Leakage Prevention**"
    ]
    
    for step in pipeline_steps:
        st.markdown(step)
    
    st.markdown("---")
    
    # Missing Value Handling
    st.subheader("1️⃣ Missing Value Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strategy Applied:**")
        strategy = preprocessing_info['missing_value_strategy']
        st.code(f"""
# Delay columns → {strategy['delay_columns']}
delay_cols = ['carrier_delay', 'weather_delay', 'nas_delay', 
              'security_delay', 'late_aircraft_delay']
df[delay_cols] = df[delay_cols].fillna(0)

# Count columns → {strategy['count_columns']}
count_cols = ['arr_flights', 'arr_del15', 'carrier_ct']
df[count_cols] = df[count_cols].fillna(0)

# Categorical → {strategy['categorical_columns']}
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        """, language='python')
    
    with col2:
        st.success("**✅ Rationale:**")
        st.write("• **Zero-fill for delays**: Missing = no delay occurred")
        st.write("• **Zero-fill for counts**: Missing = no occurrences") 
        st.write("• **Unknown for categories**: Preserves missing data info")
        st.write("• **Business logic**: Aligns with operational reality")
    
    # Outlier Treatment
    with st.expander("3️⃣ Outlier Detection & Treatment"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**IQR Method Applied:**")
            st.code("""
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    data[column] = np.clip(data[column], 
                          lower_bound, upper_bound)
            """, language='python')
        
        with col2:
            st.markdown("**Outliers Treated:**")
            outliers = preprocessing_info['outlier_treatment']['outliers_capped']
            for col, count in outliers.items():
                st.write(f"• {col}: {count:,} outliers capped")
            
            st.info("Extreme delays capped to reduce model bias while preserving distribution shape")
    
    # Feature Engineering
    st.subheader("4️⃣ Feature Engineering")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Target Variable", "📅 Temporal Features", "📊 Operational Features"])
    
    with tab1:
        target_info = preprocessing_info['feature_engineering']['target_variable']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Definition:**")
            st.code(f"""
# {target_info['definition']}
df['delay_rate'] = df['arr_del15'] / df['arr_flights'].replace(0, 1)
df['{target_info['name']}'] = (df['delay_rate'] > 0.25).astype(int)
            """, language='python')
        
        with col2:
            class_dist = dataset_stats['class_distribution']
            st.metric("Normal Operations", f"{class_dist['0']:.1%}")
            st.metric("High Delay Periods", f"{class_dist['1']:.1%}")
            st.info(target_info['threshold_rationale'])
    
    with tab2:
        st.markdown("**Time-Based Features:**")
        temporal_features = preprocessing_info['feature_engineering']['temporal_features']
        
        st.code("""
# Seasonal patterns
df['quarter'] = ((df['month'] - 1) // 3) + 1
df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
df['is_peak_travel'] = df['month'].isin([6, 7, 8, 11, 12]).astype(int)
        """, language='python')
        
        st.success("**✅ Captures:** Seasonal effects, peak travel periods, quarterly cycles")
    
    with tab3:
        st.markdown("**Operational Features:**")
        operational_features = preprocessing_info['feature_engineering']['operational_features']
        
        st.code("""
# Operational metrics
df['flights_per_day'] = df['arr_flights'] / 30
df['cancellation_rate'] = df['arr_cancelled'] / df['arr_flights'].replace(0, 1)
df['total_disruptions'] = df['arr_cancelled'] + df['arr_diverted']

# Airport & Carrier aggregations (without leakage)
airport_stats = df.groupby('airport').agg({
    'arr_flights': ['sum', 'mean'],
    'arr_cancelled': 'sum'
})
        """, language='python')
        
        st.success("**✅ Captures:** Daily operations, disruption rates, hub characteristics")
    
    # Encoding and Scaling
    st.subheader("6️⃣ & 7️⃣ Encoding & Scaling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Categorical Encoding:**")
        encoding_info = preprocessing_info['encoding_scaling']
        st.code("""
from sklearn.preprocessing import LabelEncoder

# Encode high-cardinality categories
for col in ['carrier', 'airport']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
        """, language='python')
        
        st.info(f"**Strategy:** {encoding_info['categorical_encoding']}")
    
    with col2:
        st.markdown("**Feature Scaling:**")
        st.code("""
from sklearn.preprocessing import StandardScaler

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=[np.number]).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        """, language='python')
        
        st.info(f"**Strategy:** {encoding_info['numerical_scaling']}")
    
    # Data Leakage Prevention
    with st.expander("8️⃣ Data Leakage Prevention"):
        leakage_info = preprocessing_info['data_leakage_prevention']
        
        st.warning("**⚠️ Removed Features to Prevent Leakage:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Direct Leakage (Future Info):**")
            for feature in leakage_info['removed_features'][:6]:
                st.text(f"❌ {feature}")
        
        with col2:
            st.markdown("**Derived Leakage (Target-based):**")
            for feature in leakage_info['removed_features'][6:]:
                st.text(f"❌ {feature}")
        
        st.error("These features contain information not available at prediction time or derived from target")
    
    # Final Summary
    st.subheader("📊 Preprocessing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Records", f"{dataset_stats['total_records']:,}")
    with col2:
        st.metric("Final Features", leakage_info['final_feature_count'])
    with col3:
        st.metric("Missing Values", "0")
    with col4:
        st.metric("Target Classes", "2")
    
    st.success("""
    **✅ Preprocessing Complete - Ready for ML Training:**
    
    🔹 **Data Quality**: All missing values handled, outliers treated, quality checks passed  
    🔹 **Feature Engineering**: 22 meaningful features created from domain knowledge  
    🔹 **Target Definition**: Binary classification with business-relevant threshold  
    🔹 **Encoding**: Optimal representation for tree-based algorithms  
    🔹 **Scaling**: Standardized features for algorithm compatibility  
    🔹 **Leakage Prevention**: Future information and target-derived features removed  
    
    👉 **Next**: View **Model Training** to see how preprocessing enabled effective ML.
    """)