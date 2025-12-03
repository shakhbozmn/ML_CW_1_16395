import streamlit as st
import json
import os

st.set_page_config(
    page_title="Flight Delay Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Delay Analysis Dashboard")

# Navigation
page = st.sidebar.selectbox(
    "Navigate:",
    ["🏠 Home", "📊 Data Exploration", "⚙️ Preprocessing", "🤖 Model Training", "📈 Evaluation", "🔮 Prediction"]
)

# Load exported results
@st.cache_data
def load_results():
    try:
        base_path = os.path.dirname(os.path.dirname(__file__))  # go from src/ → root
        results_path = os.path.join(base_path, "results")

        # Load all exported data
        with open(os.path.join(results_path, "model_results.json"), "r") as f:
            model_results = json.load(f)

        with open(os.path.join(results_path, "dataset_stats.json"), "r") as f:
            dataset_stats = json.load(f)
            
        with open(os.path.join(results_path, "business_insights.json"), "r") as f:
            business_insights = json.load(f)
            
        with open(os.path.join(results_path, "preprocessing_info.json"), "r") as f:
            preprocessing_info = json.load(f)
            
        with open(os.path.join(results_path, "model_analysis.json"), "r") as f:
            model_analysis = json.load(f)

        return model_results, dataset_stats, business_insights, preprocessing_info, model_analysis

    except FileNotFoundError as e:
        st.error(f"Results not found! Please run the notebook export cells first. Missing: {e}")
        return None, None, None, None, None


model_results, dataset_stats, business_insights, preprocessing_info, model_analysis = load_results()

# Simple page routing
if page == "🏠 Home":
    from pages import home
    home.show(model_results, dataset_stats, business_insights)
elif page == "📊 Data Exploration":
    from pages import exploration  
    exploration.show(dataset_stats)
elif page == "⚙️ Preprocessing":
    from pages import preprocessing
    preprocessing.show(dataset_stats, preprocessing_info)
elif page == "🤖 Model Training":
    from pages import training
    training.show(model_results)
elif page == "📈 Evaluation":
    from pages import evaluation
    evaluation.show(model_results, model_analysis)
elif page == "🔮 Prediction":
    from pages import prediction
    prediction.show()