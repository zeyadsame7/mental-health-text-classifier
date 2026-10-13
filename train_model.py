import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.utils import resample

class MentalHealthModelTrainer:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Categories for classification
        self.categories = [
            'depression',
            'anxiety',
            'suicidal_thoughts',
            'self_harm',
            'isolation',
            'stress',
            'trauma',
            'substance_abuse',
            'eating_disorder',
            'bipolar'
        ]

    def preprocess_text(self, text):
        """Clean and preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)

    def prepare_training_data(self):
        """Prepare training data from the new CSV file for multi-class classification."""
        df = pd.read_csv('your_new_dataset.csv')
        X = df['statement'].fillna('')
        y = df['status'].astype(str)
        return X, y

    def print_and_balance_data(self, X, y, balance=True):
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")
        if not balance:
            return X, y
        # Balance the dataset by undersampling all classes to the size of the smallest class
        df = pd.DataFrame({'X': X, 'y': y})
        min_count = df['y'].value_counts().min()
        df_balanced = pd.concat([
            df[df.y == label].sample(n=min_count, random_state=42)
            for label in df['y'].unique()
        ])
        df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle
        print(f"Balanced dataset: {df_balanced.y.value_counts().to_dict()}")
        return df_balanced.X, df_balanced.y

    def train(self):
        """Train the model on the prepared data."""
        # Prepare data
        X, y = self.prepare_training_data()
        # Print and balance data
        X, y = self.print_and_balance_data(X, y, balance=True)
        # Preprocess text
        X_processed = X.apply(self.preprocess_text)
        # First split: train vs temp (validation+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        # Second split: validation vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)
        # Train classifier
        print("Training classifier for multi-class mental health detection...")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train_vec, y_train)
        # Evaluate on validation set
        y_val_pred = self.classifier.predict(X_val_vec)
        print("\nValidation Set Classification Report:")
        print(classification_report(y_val, y_val_pred))
        # Evaluate on test set
        y_test_pred = self.classifier.predict(X_test_vec)
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred))
        # Save the model and vectorizer
        self.save_model()

    def save_model(self):
        """Save the trained model and vectorizer."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        joblib.dump(model_data, 'mental_health_model.joblib')
        print("\nModel saved successfully!")

    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            model_data = joblib.load('mental_health_model.joblib')
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            print("Model loaded successfully!")
            return True
        except:
            print("No trained model found.")
            return False

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    # Train the model
    trainer = MentalHealthModelTrainer()
    trainer.train() 