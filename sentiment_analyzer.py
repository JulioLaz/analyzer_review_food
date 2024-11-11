import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # Cambio a SGD por ser más ligero
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Cambio a stemming por ser más ligero
from datetime import datetime
import pytz
import pickle
import re
import string
import nltk
# from nltk.tokenize import word_tokenize
local_tz = pytz.timezone('America/Argentina/Buenos_Aires')
local_time_now = datetime.now(local_tz).strftime("%d_%b_%Y_%Hhs_%Mmin")

def download_nltk_resources():
    """Descarga recursos necesarios de NLTK"""
    resources = ['stopwords', 'punkt']  # Reducidos los recursos necesarios
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

print("Downloading NLTK resources...")
download_nltk_resources()

ngram=2
features=15000
path=f'optimized_sentiment_model_{ngram}_{features}_{local_time_now}.pkl'

class OptimizedSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=features,  
            ngram_range=(1, ngram),
            min_df=4,  # Aumentado para reducir características
            max_df=0.9,
            strip_accents='unicode',
            sublinear_tf=True  # Aplico escala logarítmica para reducir varianza
        )
        
        self.model = SGDClassifier(
            loss='modified_huber',  # Permite probabilidades
            penalty='l2',
            alpha=1e-4,
            max_iter=100,
            tol=1e-3,
            random_state=42,
            class_weight='balanced'
        )
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # Stemming en lugar de lemmatization
        # vocab = self.vectorizer.vocabulary
        # Lista reducida de palabras de negación más comunes
        self.negation_words = {'no', 'not', "n't", 'never', 'none', 'nobody', 'nowhere'}
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
            
        try:
            # Convertir a minúsculas y eliminar URLs
            text = str(text).lower()
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Tokenización simple y eficiente
            words = text.split()
            
            # Procesamiento optimizado
            processed_words = []
            negation = False
            
            for word in words:
                # Eliminar puntuación excepto apóstrofes en negaciones
                if word.endswith("n't"):
                    processed_words.append("not")
                    negation = True
                    continue
                
                # Limpiar palabra
                word = ''.join(c for c in word if c not in string.punctuation)
                if not word:
                    continue
                
                # Aplicar stemming si no es stopword ni negación
                if word not in self.stop_words or word in self.negation_words:
                    word = self.stemmer.stem(word)
                    if negation:
                        word = 'NEG_' + word
                        negation = False
                    processed_words.append(word)
                
                if word in self.negation_words:
                    negation = True
            
            return ' '.join(processed_words)
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return str(text)

    def train(self, data_path, sample_size=None):
        try:
            print("Loading data...")
            df = pd.read_csv(data_path)
            
            # Opcionalmente usar solo una muestra de los datos
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            df = df[['Score', 'Text']]
            
            print("Converting scores...")
            df = df[df['Score'] != 3]
            df['sentiment'] = (df['Score'] >= 4).astype(int)
            
            print("Preprocessing texts...")
            df['processed_text'] = df['Text'].apply(self.preprocess_text)
            
            print("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df['sentiment'],
                test_size=0.2,
                random_state=42
            )
            
            print("Vectorizing and training...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_train_vec=X_train_vec.astype(np.float32)
            self.model.fit(X_train_vec, y_train)
            
            # Evaluación rápida
            X_test_vec = self.vectorizer.transform(X_test)
            X_test_vec=X_test_vec.astype(np.float32)
            accuracy = self.model.score(X_test_vec, y_test)
            print(f"\nModel Accuracy: {accuracy:.2f}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(self, text):
        try:
            processed = self.preprocess_text(text)
            vec_text = self.vectorizer.transform([processed])
            
            # Obtener probabilidades
            proba = self.model.predict_proba(vec_text)[0]
            sentiment = "Negative" if proba[0] > proba[1] else "Positive"
            
            return {
                'sentiment': sentiment,
                'confidence': float(max(proba) * 100),
                'probabilities': {
                    'negative': float(proba[0]),
                    'positive': float(proba[1])
                }
            }
        except Exception as e:
            return {'error': str(e), 'sentiment': 'unknown'}

    def save_model(self, model_path=path):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'vocabulary': self.vectorizer.vocabulary  # Incluir vocabulario
            }, f, protocol=4)
    
    @classmethod
    def load_model(cls, model_path):
        analyzer = cls()
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
            analyzer.vectorizer = components['vectorizer']
            analyzer.model = components['model']
            analyzer.vocab = components.get('vocabulary', analyzer.vectorizer.vocabulary)  # Cargar vocabulario

        return analyzer

if __name__ == "__main__":
    try:
        print("Initializing analyzer...")
        analyzer = OptimizedSentimentAnalyzer()
        
        # Usar solo una muestra de los datos para entrenamiento
        print("\nStarting training...")
        analyzer.train('Reviews.csv', sample_size=50000)  # Ajusta este número según necesites
        
        print("\nSaving model...")
        analyzer.save_model(path)
        
        # Prueba rápida
        test_texts = [
            "i think it was not good",
            "this product is excellent",
            "terrible experience, would not recommend"
        ]
        
        print("\nTesting predictions:")
        for text in test_texts:
            result = analyzer.predict(text)
            print(f"\nText: '{text}'")
            print(f"Sentiment: {result['sentiment']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.1f}%")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")