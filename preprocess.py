import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and preprocess text for sentiment analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def stem_words(self, tokens):
        """
        Apply stemming to tokens
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Stem words
        stemmed_tokens = self.stem_words(filtered_tokens)
        
        return ' '.join(stemmed_tokens)

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_text = "I absolutely love this product! It's amazing and works perfectly."
    processed = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
