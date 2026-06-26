from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_model import SentimentAnalyzer
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global analyzer instance
analyzer = None

def initialize_model():
    """Initialize the sentiment analyzer with trained model"""
    global analyzer
    model_path = 'models/model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            analyzer = SentimentAnalyzer(model_path, vectorizer_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to training mode...")
            train_model()
    else:
        print("Model not found. Training new model...")
        train_model()

def train_model():
    """Train model if not exists"""
    global analyzer
    from train_model import main
    main()  # This will train and save the model
    analyzer = SentimentAnalyzer('models/model.pkl', 'models/vectorizer.pkl')
    print("Model trained and loaded!")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'endpoints': {
            '/predict': 'POST - Analyze sentiment of text(s)',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for provided text(s)
    Expected JSON:
    {
        "text": "I love this product!"  (for single text)
        OR
        {
            "texts": ["I love this!", "I hate this!"]
        }
    """
    try:
        if not analyzer:
            return jsonify({'error': 'Model not initialized'}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Check if single text or multiple texts
        if 'text' in data:
            # Single text prediction
            text = data['text'].strip()
            if not text:
                return jsonify({'error': 'Text cannot be empty'}), 400
            
            result = analyzer.predict_single(text)
            return jsonify({
                'success': True,
                'result': result
            })
        
        elif 'texts' in data:
            # Multiple texts prediction
            texts = [t.strip() for t in data['texts'] if t.strip()]
            if not texts:
                return jsonify({'error': 'Texts list cannot be empty'}), 400
            
            results = analyzer.predict(texts)
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results)
            })
        
        else:
            return jsonify({
                'error': 'Invalid request. Provide "text" or "texts" in JSON'
            }), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Expected JSON:
    {
        "texts": ["text1", "text2", "..."]
    }
    """
    try:
        if not analyzer:
            return jsonify({'error': 'Model not initialized'}), 503
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400
        
        texts = [t.strip() for t in data['texts'] if t.strip()]
        if not texts:
            return jsonify({'error': 'Texts list cannot be empty'}), 400
        
        results = analyzer.predict(texts)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Sentiment Analysis API Server")
    print("=" * 60)
    
    # Initialize model on startup
    initialize_model()
    
    print("\nStarting Flask server...")
    print("API available at: http://localhost:5000")
    print("To test: POST /predict with JSON data")
    print("=" * 60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
