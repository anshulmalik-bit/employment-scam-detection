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
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_preprocessor.vectorizer, 'text'), 
            ('metadata', 'passthrough', ['has_company_logo', 'has_questions', 'telecommuting'])
        ],
        remainder='drop'
    )
    
    # Clean text columns before passing them into the transformer pipeline
    print("Cleaning text data for TFIDF...")
    X = X.copy()
    X['text'] = [text_preprocessor.clean_text(doc) for doc in X['text']]

    print("Splitting dataset into 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Fit the ColumnTransformer on training data
    print("Fitting ColumnTransformer (TF-IDF + Metadata)...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # 3. Apply SMOTE on the ALREADY-TRANSFORMED training data
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    print(f"  Before SMOTE: {len(y_train)} samples | After SMOTE: {len(y_train_resampled)} samples")
    
    # 4. Train the Random Forest on the resampled data
    print("Training Random Forest Classifier (with balanced class weights)...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    print("Evaluating Random Forest (V2 with SMOTE & Metadata):")
    y_pred_rf = rf_model.predict(X_test_transformed)
    print(classification_report(y_test, y_pred_rf))
    
    # 5. Save the components SEPARATELY (no imblearn in saved artifacts!)
    # This way the app only needs scikit-learn to load them
    print("Saving model components separately for cloud compatibility...")
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'column_transformer.joblib'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model_v2.joblib'))
    joblib.dump(text_preprocessor, os.path.join(MODELS_DIR, 'text_cleaner.joblib'))
    print(f"All artifacts saved in {MODELS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
