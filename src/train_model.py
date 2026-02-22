import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from data_loader import load_data
from preprocessing import TextPreprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

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
    
    X = df['text'].values
    y = df['fraudulent'].values
    
    print("Preprocessing text and computing TF-IDF...")
    preprocessor = TextPreprocessor(max_features=5000)
    X_tfidf = preprocessor.fit_transform(X)
    
    print("Splitting dataset into 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Logistic Regression with class_weight='balanced' to handle imbalance
    print("Training Logistic Regression Model (with balanced class weights)...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    print("Evaluating Logistic Regression:")
    y_pred_lr = lr_model.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier (with balanced class weights)...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Evaluating Random Forest:")
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    # We will save the Random Forest model and the Vectorizer
    print("Saving Random Forest model and Preprocessor pipeline...")
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    print(f"Artifacts saved in {MODELS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
