import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import TextPreprocessor

class SentimentAnalyzer:
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize sentiment analyzer with trained model
        """
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the sentiment analysis model
        """
        # Preprocess training data
        X_train_processed = [self.preprocessor.preprocess(text) for text in X_train]
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Train the model
        self.pipeline.fit(X_train_processed, y_train)
        
        # Store components for easy access
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.model = self.pipeline.named_steps['classifier']
        
        return self
    
    def predict(self, texts):
        """
        Predict sentiment for a list of texts
        """
        if not self.pipeline:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Get predictions
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        results = []
        for i, text in enumerate(texts):
            sentiment = "Positive" if predictions[i] == 1 else "Negative"
            confidence = max(probabilities[i]) * 100
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': f"{confidence:.2f}%",
                'raw_confidence': float(max(probabilities[i]))
            })
        
        return results
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text
        """
        results = self.predict([text])
        return results[0]
    
    def save_model(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """
        Save trained model and vectorizer
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """
        Load trained model and vectorizer
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Recreate pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")

# Example usage
if __name__ == "__main__":
    # Sample training data
    sample_texts = [
        "I love this product! It's amazing.",
        "This is terrible, I hate it.",
        "Good quality, works well.",
        "Bad experience, would not recommend.",
        "Excellent service and fast delivery.",
        "Waste of money, very disappointed."
    ]
    sample_labels = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative
    
    # Train model
    analyzer = SentimentAnalyzer()
    analyzer.train(sample_texts, sample_labels)
    
    # Test prediction
    test_text = "This is a fantastic product, I'm very happy with it."
    result = analyzer.predict_single(test_text)
    print(f"Test Result: {result}")
    
    # Save model
    analyzer.save_model()
