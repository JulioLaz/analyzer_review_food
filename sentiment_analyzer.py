import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
from nltk.corpus import stopwords
import nltk
import string

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Convertir a minúsculas
        text = text.lower()
        # Remover puntuación
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Identificar palabras de negación
        negations = ["not", "no", "never", "cannot"]
        words = text.split()
        for i in range(len(words) - 1):
            if words[i] in negations:
                words[i] = f"{words[i]}_{words[i+1]}"
                words[i+1] = ''
        # Remover stopwords y palabras vacías después de unir negaciones
        text = ' '.join([word for word in words if word not in self.stop_words and word])
        # Remover URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return text

# class SentimentAnalyzer:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
#         self.model = LogisticRegression()
#         self.stop_words = set(stopwords.words('english'))
        
#     def preprocess_text(self, text):
#         # Convertir a minúsculas
#         text = text.lower()
#         # Remover puntuación
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         # Remover stopwords
#         text = ' '.join([word for word in text.split() if word not in self.stop_words])
#         # Remover URLs
#         text = re.sub(r'https?://\S+|www\.\S+', '', text)
#         return text
        
    def train(self, data_path):
        # Cargar datos
        df = pd.read_csv(data_path)
        df = df[['Score', 'Text']]
        
        # Convertir scores a sentimiento binario (1-2: negativo, 4-5: positivo)
        df = df[df['Score'] != 3]  # Removemos neutral
        df['sentiment'] = (df['Score'] >= 4).astype(int)
        
        # Preprocesar textos
        df['processed_text'] = df['Text'].apply(self.preprocess_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorizar textos
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Entrenar modelo
        self.model.fit(X_train_vec, y_train)
        
        # Evaluar modelo
        X_test_vec = self.vectorizer.transform(X_test)
        accuracy = self.model.score(X_test_vec, y_test)
        print(f"Model accuracy: {accuracy}")
        
        return accuracy

    def predict(self, text):
        # Preprocesar el texto
        processed = self.preprocess_text(text)
        
        # Filtrar palabras que están en el vocabulario
        words = processed.split()
        valid_words = [word for word in words if word in self.vectorizer.vocabulary_]
        
        # Si no quedan palabras válidas, retornar un mensaje
        if not valid_words:
            return {
                'error': 'No valid words for prediction',
                'sentiment': 'unknown',
                'negative': None,
                'positive': None
            }
        
        # Vectorizar solo las palabras válidas
        vec_text = self.vectorizer.transform([' '.join(valid_words)])
        
        # Realizar la predicción
        prediction = self.model.predict_proba(vec_text)[0]
        
        return {
            'negative': float(prediction[0]),
            'positive': float(prediction[1])
        }



    def predict_00(self, text):
        # Preprocesar texto
        processed = self.preprocess_text(text)
        # Vectorizar
        vec_text = self.vectorizer.transform([processed])
        # Predecir
        prediction = self.model.predict_proba(vec_text)[0]
        return {
            'negative': float(prediction[0]),
            'positive': float(prediction[1])
        }
    
    def save_model(self, model_path='sentiment_model.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
    
    @classmethod
    def load_model_00(cls, model_path):
        analyzer = cls()
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
            analyzer.vectorizer = components['vectorizer']
            analyzer.model = components['model']
        return analyzer
    
    @classmethod
    def load_model(cls, model_path):
        analyzer = cls()
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
            analyzer.vectorizer = components['vectorizer']
            analyzer.model = components['model']
            analyzer.vocab = components.get('vocabulary')  # Cargar vocabulario
            analyzer.vectorizer.vocabulary_ = analyzer.vocab
        return analyzer

# Ejemplo de uso para entrenamiento
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.train('Reviews.csv')
    analyzer.save_model('sentiment_model.pkl')