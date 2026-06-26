import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentiment_model import SentimentAnalyzer
import nltk
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_imdb_dataset():
    """
    Load IMDB dataset or create sample data if file doesn't exist
    """
    try:
        # Try to load IMDB dataset (you would need to download this)
        # For this example, we'll create a sample dataset
        sample_data = pd.DataFrame({
            'review': [
                "This movie was fantastic! Great acting and amazing story.",
                "Terrible film, worst I've ever seen. Waste of time.",
                "Good movie with excellent performances. Would watch again.",
                "Boring and predictable. Not worth watching.",
                "Amazing cinematography and brilliant direction. A must watch.",
                "Disappointing. The plot was weak and characters were flat.",
                "Excellent film! Highly recommend to everyone.",
                "Horrible acting and terrible script. Avoid this movie.",
                "Beautiful story with great emotions. Loved it.",
                "Waste of money. Poor quality and boring plot.",
                "Loved every moment of this film. It was perfect.",
                "Hated it. Complete waste of my time.",
                "Good movie overall, decent acting and nice story.",
                "Not my cup of tea. Found it quite boring.",
                "Fantastic movie, one of the best I've seen this year.",
                "Poorly made film with no real story. Very disappointing.",
                "Great movie, loved the characters and the plot.",
                "Awful film, wouldn't recommend to anyone.",
                "Wonderful movie with a beautiful message.",
                "Terrible experience. Bad acting and worst plot."
            ],
            'sentiment': [
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0
            ]
        })
        print("Using sample dataset (20 examples)")
        return sample_data['review'].tolist(), sample_data['sentiment'].tolist()
    
    except FileNotFoundError:
        print("IMDB dataset not found. Using sample data.")
        return sample_data['review'].tolist(), sample_data['sentiment'].tolist()

def main():
    print("=" * 60)
    print("Sentiment Analysis Model Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading dataset...")
    texts, labels = load_imdb_dataset()
    print(f"Loaded {len(texts)} samples")
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    analyzer = SentimentAnalyzer()
    analyzer.train(X_train, y_train)
    print("Training completed!")
    
    # Evaluate on test set
    print("\nEvaluating model...")
    results = analyzer.predict(X_test)
    y_pred = [1 if r['sentiment'] == 'Positive' else 0 for r in results]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    analyzer.save_model()
    
    # Test with custom examples
    print("\n" + "=" * 60)
    print("Testing with custom examples:")
    print("=" * 60)
    
    test_examples = [
        "I absolutely love this product! It's the best purchase I've ever made.",
        "This is the worst experience ever. I'm very disappointed.",
        "The movie was okay, nothing special but not terrible either.",
        "Amazing service! Fast shipping and great quality.",
        "Complete waste of money. Would not recommend to anyone.",
        "Good value for money. Satisfied with the purchase."
    ]
    
    for example in test_examples:
        result = analyzer.predict_single(example)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved to 'models/' directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
