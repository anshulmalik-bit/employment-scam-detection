import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data (only downloads if not already present)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    def __init__(self, max_features=5000, **kwargs):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
        
    def clean_text(self, text):
        """ Lowercase, remove punctuation and numbers, tokenize, lemmatize, and remove stop words """
        if not isinstance(text, str):
            text = str(text)
            
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize (using simple split to avoid missing punkt issues if any, but nltk word_tokenize is better)
        try:
             tokens = nltk.word_tokenize(text)
        except LookupError:
             tokens = text.split()
            
        # Remove stop words and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return " ".join(cleaned_tokens)
        
    def fit_transform(self, X):
        """ Clean text and apply TF-IDF """
        print("Cleaning text data...")
        # Clean each document
        cleaned_X = [self.clean_text(doc) for doc in X]
        print("Applying TF-IDF vectorization...")
        # Fit and transform
        X_tfidf = self.vectorizer.fit_transform(cleaned_X)
        return X_tfidf
        
    def transform(self, X):
        """ Clean text and transform using fitted TF-IDF """
        cleaned_X = [self.clean_text(doc) for doc in X]
        return self.vectorizer.transform(cleaned_X)

if __name__ == "__main__":
    # Simple test
    sample_data = [
        "This is an AMAZING job opportunity! Click http://fake.com to apply.",
        "We are looking for a Software Engineer with 5 years experience."
    ]
    preprocessor = TextPreprocessor(max_features=10)
    tfidf_matrix = preprocessor.fit_transform(sample_data)
    print("Features extracted:", preprocessor.vectorizer.get_feature_names_out())
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
