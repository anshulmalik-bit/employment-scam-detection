import os
import joblib
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.joblib')

# In order to unpickle the Custom TextPreprocessor, its class needs to be defined or imported
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from preprocessing import TextPreprocessor
except ImportError:
    st.error("TextPreprocessor module could not be loaded. Ensure the 'src' directory exists and contains 'preprocessing.py'.")

# Set page config
st.set_page_config(
    page_title="Employment Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .high-risk {
        color: white;
        background-color: #ff4b4b;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    .low-risk {
        color: white;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    .keyword-highlight {
        background-color: #ffcccb;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🛡️ Algorithmic Detection of Fraudulent Employment Opportunities")
st.markdown("""
This platform uses a Natural Language Processing (NLP) classification engine to analyze unstructured job descriptions and predict the probability of fraud.
Prepared for: **Preeti Mam** | Course: **Business Analysis using Python**
""")

st.divider()

# Load Models
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

if model is None or preprocessor is None:
    st.warning("⚠️ The classification model is not found. Please train the model first by running `python src/train_model.py`.")
else:
    st.success("✅ Model and NLP Pipeline loaded successfully.")
    
    st.header("Analyze Job Posting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g. Data Entry Clerk")
        company_profile = st.text_area("Company Profile", placeholder="Enter the company profile or background information here...", height=100)
        job_description = st.text_area("Job Description", placeholder="Paste the full unstructured job description here...", height=200)
        
        analyze_btn = st.button("Analyze for Fraud")
        
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_btn:
            if not job_title and not job_description:
                st.error("Please provide at least a Job Title or Description.")
            else:
                with st.spinner("Analyzing text using NLP Pipeline..."):
                    # Combine text similarly to training
                    combined_text = f"{job_title} {company_profile} {job_description}"
                    
                    # Transform text
                    X_input = preprocessor.transform([combined_text])
                    
                    # Predict
                    probability = model.predict_proba(X_input)[0][1] # Probability of Class 1 (Fraud)
                    prediction = model.predict(X_input)[0]
                    
                    st.metric(label="Fraud Probability Score", value=f"{probability * 100:.1f}%")
                    
                    if prediction == 1:
                        st.markdown("<div class='high-risk'><h3>High Risk of Fraud 🚨</h3>This job posting exhibits patterns commonly found in scams.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='low-risk'><h3>Low Risk ✅</h3>This job posting appears legitimate based on our model.</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown("#### Suspicious Feature Analysis")
                    st.info("The model analyzed the TF-IDF feature space to determine this score. Keywords highly associated with scams in the training data contributed to this probability.")
                    
                    # Extract top features
                    # This is a simplified keyword highlighter (in a real scenario we could use LIME or SHAP)
                    feature_names = preprocessor.vectorizer.get_feature_names_out()
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        # Find indices of non-zero TF-IDF values for this instance
                        nonzero_indices = X_input[0].nonzero()[1]
                        
                        top_keywords = []
                        for idx in nonzero_indices:
                            top_keywords.append((feature_names[idx], importances[idx]))
                            
                        # Sort by importance
                        top_keywords.sort(key=lambda x: x[1], reverse=True)
                        
                        st.write("Top detected keywords influencing the model:")
                        for kw, imp in top_keywords[:5]:
                            st.markdown(f"- <span class='keyword-highlight'>{kw}</span> (Importance: {imp:.4f})", unsafe_allow_html=True)

st.sidebar.title("Business Value")
st.sidebar.markdown("""
### Why this matters
The proliferation of fake job listings damages employer branding, defrauds job seekers, and creates massive liability for job boards.

### Our Approach
By automating trust and safety using data, we reduce the need for manual human review and scale moderation instantly.

### Tech Stack
- **Data Manipulation**: pandas, numpy
- **NLP Pipeline**: nltk, scikit-learn (TfidfVectorizer)
- **Predictive Modeling**: scikit-learn (RandomForest)
- **Dashboard**: Streamlit
""")
