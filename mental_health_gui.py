import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import numpy as np

class MultiClassMentalHealthDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Load trained model
        try:
            model_data = joblib.load('mental_health_model.joblib')
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            print("Model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"No trained model found. Error: {e}")
            self.model_loaded = False

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def analyze_text(self, text):
        processed_text = self.preprocess_text(text)
        if self.model_loaded:
            text_vec = self.vectorizer.transform([processed_text])
            proba = self.classifier.predict_proba(text_vec)[0]
            classes = self.classifier.classes_
            top_idx = np.argmax(proba)
            top_class = classes[top_idx]
            return {
                'category': top_class
            }
        else:
            return {
                'category': 'Unknown'
            }

class MultiClassMentalHealthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Health Category Detector")
        self.root.geometry("600x250")
        self.detector = MultiClassMentalHealthDetector()

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Enter chat message to analyze:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.text_input = scrolledtext.ScrolledText(main_frame, width=60, height=6)
        self.text_input.grid(row=1, column=0, pady=(0, 10))

        ttk.Button(main_frame, text="Analyze", command=self.analyze_text).grid(row=2, column=0, pady=(0, 10))

        results_frame = ttk.LabelFrame(main_frame, text="Analysis Result", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(results_frame, text="Predicted Status:").grid(row=0, column=0, sticky=tk.W)
        self.category_label = ttk.Label(results_frame, text="")
        self.category_label.grid(row=0, column=1, sticky=tk.W)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)

    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            self.show_error("Please enter some text to analyze")
            return
        try:
            result = self.detector.analyze_text(text)
            category = result['category']
            self.category_label.config(text=category)
        except Exception as e:
            self.show_error(f"Error analyzing text: {str(e)}")

    def show_error(self, message):
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    root = tk.Tk()
    app = MultiClassMentalHealthGUI(root)
    root.mainloop() 