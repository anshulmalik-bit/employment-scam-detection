import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from data_loader import load_data
from preprocessing import TextPreprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# A helper wrapper to extract text for the preprocessor
def get_text_col(X):
    return X['text'].values

def train_and_evaluate():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("Loading data...")
    df = load_data()
    
    # Check if target variable exists
    if 'fraudulent' not in df.columns:
        print("Error: Target variable 'fraudulent' not found in dataset. Ensure correct dataset is loaded.")
        return
        
    # Drop rows without text or target
    df = df.dropna(subset=['text', 'fraudulent'])
    
    # Features and Target
    X = df[['text', 'has_company_logo', 'has_questions', 'telecommuting']]
    y = df['fraudulent'].values
    
    # 1. Create the Preprocessor ColumnTransformer
    print("Setting up Text and Metadata ColumnTransformer...")
    # Initialize the custom text preprocessor with N-Grams
    text_preprocessor = TextPreprocessor(max_features=5000, ngram_range=(1, 2))
    
    # Column transformer to apply TF-IDF to 'text', and pass through the metadata columns unchanged
    # We must wrap text transformation in a pipeline because ColumnTransformer expects fit_transform
    # to be called on a 1D/2D array, and TextPreprocessor expects a list of strings
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_preprocessor.vectorizer, 'text'), 
            ('metadata', 'passthrough', ['has_company_logo', 'has_questions', 'telecommuting'])
        ],
        remainder='drop'
    )
    
    # Clean text columns before passing them into the transformer pipeline
    print("Cleaning text data for TFIDF...")
    X['text'] = [text_preprocessor.clean_text(doc) for doc in X['text']]

    print("Splitting dataset into 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Build the imblearn Pipeline with SMOTE and the Classifier
    print("Building Random Forest Pipeline with SMOTE...")
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    # Train the Random Forest Model Pipeline
    print("Training Random Forest Pipeline...")
    rf_pipeline.fit(X_train, y_train)
    
    print("Evaluating Random Forest Pipeline (V2 with SMOTE & Metadata):")
    y_pred_rf = rf_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    # Print feature names to verify N-Grams and columns
    print("\nExtracting feature names out...")
    try:
        tfidf_features = rf_pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
        all_features = list(tfidf_features) + ['has_company_logo', 'has_questions', 'telecommuting']
        print(f"Total features extracted: {len(all_features)} (including N-Grams and metadata)")
    except Exception as e:
        print("Note: Could not extract specific feature names for debugging:", e)
        
    # We will save the entire fitted pipeline because it contains the text vectorizer, metadata passthrough, and model
    print("Saving the full Random Forest Pipeline (Model V2)...")
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, 'rf_pipeline_v2.joblib'))
    joblib.dump(text_preprocessor, os.path.join(MODELS_DIR, 'text_cleaner.joblib'))
    print(f"Pipeline and artifacts saved in {MODELS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
