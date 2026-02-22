import os
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'fake_job_postings.csv')
DATA_URL = "https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv"

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(DATA_FILE, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists locally.")

def load_data():
    """Loads the dataset and performs initial basic cleaning."""
    download_data()
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Handle missing values
    # For text columns, fill NaN with empty strings
    text_cols = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
            
    # Determine the target variable
    # EMSCAD uses 'fraudulent' column (0 for real, 1 for fake)
    if 'fraudulent' in df.columns:
        print("Target variable 'fraudulent' found.")
    else:
        print("Warning: Target variable 'fraudulent' not found!")
    
    # Isolate relevant unstructured text columns
    # We will combine 'title', 'company_profile', and 'description' as main textual features
    df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description']
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df['fraudulent'].value_counts())
