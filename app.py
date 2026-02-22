import os
import joblib
import streamlit as st
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'rf_pipeline_v2.joblib')

# Set page config
st.set_page_config(
    page_title="Employment Scam Detection V2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI including Dynamic Threshold Zones
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
        width: 100%;
    }
    .high-risk {
        color: white;
        background-color: #D32F2F; /* Red */
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    .amber-risk {
        color: black;
        background-color: #FFC107; /* Amber/Yellow */
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    .low-risk {
        color: white;
        background-color: #388E3C; /* Green */
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🛡️ NLP Fraud Detection Engine (V2 - Robust Edition)")
st.markdown("""
This upgraded classification engine combines **N-Gram NLP Analysis** with **Structural Metadata** to identify fake employment opportunities using Dynamic Risk Thresholds.
Prepared for: **Preeti Mam** | Course: **Business Analysis using Python**
""")

st.divider()

# Load Model Pipeline
@st.cache_resource
def load_artifacts():
    if not os.path.exists(PIPELINE_PATH):
        return None
    
    # Needs the TextPreprocessor dynamically in context
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    try:
        from preprocessing import TextPreprocessor
        # Re-register into global so pickle finds it
        sys.modules['preprocessing'] = sys.modules['preprocessing']
    except ImportError:
        pass
        
    pipeline = joblib.load(PIPELINE_PATH)
    return pipeline

pipeline = load_artifacts()

if pipeline is None:
    st.error("⚠️ Model Pipeline V2 not found. Please train the robust model by running `python src/train_model.py`.")
else:
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("1. Textual Analysis")
        job_title = st.text_input("Job Title", placeholder="e.g. Data Entry Clerk")
        company_profile = st.text_area("Company Profile", placeholder="Enter the company profile or background information here...", height=100)
        job_description = st.text_area("Job Description", placeholder="Paste the full unstructured job description here...", height=200)
        
        st.subheader("2. Structural Metadata")
        st.markdown("Legitimate postings often contain verifiable infrastructure. Check any that apply:")
        has_logo = st.checkbox("Has Company Logo Uploaded")
        has_questions = st.checkbox("Has Screening Questions attached")
        is_telecommuting = st.checkbox("Is a Telecommuting / Work-From-Home Position")
        
        analyze_btn = st.button("Evaluate Job Posting")
        
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_btn:
            if not job_title and not job_description:
                st.error("Please provide at least a Job Title or Description.")
            else:
                with st.spinner("Processing through ColumnTransformer Pipeline..."):
                    # Combine text similarly to training
                    combined_text = f"{job_title} {company_profile} {job_description}"
                    
                    # Create Pandas DataFrame matching training features
                    input_df = pd.DataFrame([{
                        'text': combined_text,
                        'has_company_logo': 1 if has_logo else 0,
                        'has_questions': 1 if has_questions else 0,
                        'telecommuting': 1 if is_telecommuting else 0
                    }])
                    
                    # Predict using full pipeline
                    probability = pipeline.predict_proba(input_df)[0][1] # Probability of Class 1 (Fraud)
                    prob_percentage = probability * 100
                    
                    st.metric(label="Calculated Fraud Probability", value=f"{prob_percentage:.1f}%")
                    
                    # DYNAMIC RISK CALIBRATION
                    if prob_percentage > 50.0:
                        st.markdown(
                            "<div class='high-risk'><h3>High Risk 🚨</h3>The systemic footprint of this posting strongly correlates with known scams.</div>", 
                            unsafe_allow_html=True
                        )
                    elif prob_percentage >= 20.0:
                        st.markdown(
                            "<div class='amber-risk'><h3>Caution Zone ⚠️</h3>This posting exhibits some suspicious markers. Proceed with caution and verify the employer off-platform.</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div class='low-risk'><h3>Safe ✅</h3>This posting structure matches legitimate employment patterns.</div>", 
                            unsafe_allow_html=True
                        )
                    
                    st.divider()
                    st.markdown("#### The Calibration Engine")
                    st.info(
                        "Unlike V1's binary 'Good/Bad' evaluation, this engine is optimized for high recall while mitigating false flags. " 
                        "The presence of missing company metadata acting as a multiplier on suspicious NLP N-Grams."
                    )

st.sidebar.title("Business Value (V2 Architecture)")
st.sidebar.markdown("""
### Strategic Overhaul
**The problem with early models is the trade-off between Precision and Recall.** 
A model that blindly labels anything suspicious as "Fake" produces *False Flags*, irritating real recruiters. 

### The Multi-Layered Defense
This robust V2 pipeline mitigates blind spots using:
1. **Metadata Cross-Verification**: A 40% inherently suspicious text becomes a 90% risk if the poster proves they are too lazy to upload a company logo or vetting questions.
2. **N-Grams**: Identifying context traps (pairs of words) instead of isolated flags.
3. **SMOTE**: We used synthetic minority oversampling to give the Random Forest more "villains" to learn from during training.
4. **Dynamic Risk Tiers**: Instead of a hard 50% cutoff, we introduce an "Amber Zone" (20-50%) to alert users without outright banning the poster.
""")
