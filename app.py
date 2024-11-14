from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import time
from sentiment_analyzer import OptimizedSentimentAnalyzer
import gemini
import gemini_frases
app = Flask(__name__)

@app.route('/create_phrases') 

def create_phrases(): 
   phrases = gemini_frases.chat() 
   return jsonify({'phrases': phrases})

analyzer = OptimizedSentimentAnalyzer.load_model('optimized_sentiment_model_2_20000_11_Nov_2024_20hs_11min.pkl')
vocab = analyzer.vectorizer.vocabulary_

def classify_with_stars(value):
    if not 0 <= value <= 100:
        return None  # Return None for values outside the range
    if value <= 20: stars = 1 
    elif value <= 40: stars = 2 
    elif value <= 60: stars = 3 
    elif value <= 80: stars = 4 
    else: stars = 5
    ice=5-stars
    return "⭐" * stars + "❄️"* ice

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    
    if request.method == 'POST':
        start_time = time.time()
        
        # Get the text from the form
        text = request.form.get('text', '')
        
        # Translate text
        try:
            translated_text = GoogleTranslator(source='auto', target='es').translate(text)
            print(f"Translation translated_text: {translated_text}")
        except Exception as e:
            translated_text = "Sorry, I'm having trouble translating this text."
        
        # Get Gemini analysis
      #   analyzer_gemini = '40% positive, 60% negative'  # Replace with actual gemini.chat(text)
      #   analyzer_gemini = 'Lo siento, tengo problemas para traducir este texto.'
        analyzer_gemini = gemini.chat(text)

        print('analyzer_gemini:', analyzer_gemini)
        
        # Filter text based on vocabulary
        filtered_text = " ".join(word for word in text.split() if word in vocab)
        print("Texto filtrado según vocabulario:", filtered_text)
        
        # Process Gemini results
        none_gemini = True
        if "positive" in analyzer_gemini or "negative" in analyzer_gemini:
            parts = analyzer_gemini.split(',')
            positive_gemini = int(parts[0].split('%')[0].strip())
            negative_gemini = int(parts[1].split('%')[0].strip())
            start_genimi= classify_with_stars(positive_gemini)
        else:
            none_gemini = 'None'
            positive_gemini = 0
            negative_gemini = 0
            start_genimi='No data'
        
        # Prepare response data
        if filtered_text:
            result = analyzer.predict(filtered_text)
            print('result:', result['probabilities']['positive'])
            start_sgd= classify_with_stars(result['probabilities']['positive']*100)

            response_data = {
                'filtered_text': filtered_text,
                'result': result,
                'positive': result['probabilities']['positive'],
                'negative': result['probabilities']['negative'],
                'positive_gemini': positive_gemini,
                'negative_gemini': negative_gemini,
                'translated_text': translated_text,
                'gemini_result': analyzer_gemini,
                'sentiment': 'unknown',
                'none_gemini': none_gemini,
                'start_genimi':start_genimi,
                'start_sgd':start_sgd
            }
        else:
            print('filtered_text:', filtered_text)
            start_sgd= 'No data'
            response_data = {
                'positive_gemini': positive_gemini,
                'negative_gemini': negative_gemini,
                'result': False,
                'translated_text': translated_text,
                'filtered_text': filtered_text,
                'error': 'No valid words for prediction',
                'sentiment': 'unknown',
                'positive': 0,
                'negative': 0,
                'start_sgd': start_sgd,
                'start_genimi': start_genimi,
                'gemini_result': analyzer_gemini
            }
        
        # Calculate latency
        end_time = time.time()
        response_data['latency'] = round(end_time - start_time, 3)
        print('Latencia:', response_data)
        
        return jsonify(response_data)
    
    return render_template('analyze.html')

if __name__ == '__main__':
    app.run(debug=True)