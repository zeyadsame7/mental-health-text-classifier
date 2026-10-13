import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import re

class MentalHealthDetector:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Keywords and patterns for mental health detection
        self.mental_health_keywords = {
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'tired', 'exhausted'],
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stress', 'overwhelmed'],
            'suicidal': ['suicide', 'kill', 'die', 'end', 'pain', 'suffering', 'hopeless'],
            'self_harm': ['cut', 'hurt', 'pain', 'blood', 'scars', 'burn'],
            'isolation': ['alone', 'lonely', 'isolated', 'no one', 'nobody', 'empty']
        }
        
        # Risk level thresholds
        self.risk_levels = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

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
        
        return tokens

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the text."""
        result = self.sentiment_analyzer(text)[0]
        return {
            'label': result['label'],
            'score': result['score']
        }

    def detect_keywords(self, tokens):
        """Detect mental health related keywords in the text."""
        detected_categories = {}
        
        for category, keywords in self.mental_health_keywords.items():
            matches = [word for word in tokens if word in keywords]
            if matches:
                detected_categories[category] = matches
        
        return detected_categories

    def calculate_risk_score(self, sentiment_score, keyword_matches):
        """Calculate the overall risk score."""
        # Base score from sentiment (negative sentiment increases risk)
        base_score = 1 - sentiment_score if sentiment_score > 0.5 else sentiment_score
        
        # Adjust score based on keyword matches
        keyword_score = len(keyword_matches) * 0.1
        
        # Combine scores (weighted average)
        final_score = (base_score * 0.6) + (keyword_score * 0.4)
        
        return min(final_score, 1.0)

    def determine_risk_level(self, risk_score):
        """Determine the risk level based on the risk score."""
        if risk_score >= self.risk_levels['high']:
            return 'high'
        elif risk_score >= self.risk_levels['medium']:
            return 'medium'
        else:
            return 'low'

    def analyze_text(self, text):
        """Main method to analyze text for mental health concerns."""
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Detect keywords
        keyword_matches = self.detect_keywords(tokens)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(
            sentiment['score'] if sentiment['label'] == 'POSITIVE' else 1 - sentiment['score'],
            keyword_matches
        )
        
        # Determine risk level
        risk_level = self.determine_risk_level(risk_score)
        
        return {
            'sentiment': sentiment,
            'detected_categories': keyword_matches,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommendations': self.get_recommendations(risk_level)
        }

    def get_recommendations(self, risk_level):
        """Get recommendations based on risk level."""
        recommendations = {
            'low': [
                "Continue monitoring the situation",
                "Maintain open communication",
                "Practice self-care and stress management"
            ],
            'medium': [
                "Consider professional consultation",
                "Increase monitoring frequency",
                "Document concerning patterns",
                "Maintain regular check-ins"
            ],
            'high': [
                "Immediate professional intervention recommended",
                "Contact mental health crisis hotline if needed",
                "Ensure immediate support is available",
                "Document all concerning behaviors"
            ]
        }
        return recommendations.get(risk_level, []) 