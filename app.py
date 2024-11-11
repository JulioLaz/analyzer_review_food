from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import time
from sentiment_analyzer import OptimizedSentimentAnalyzer
import gemini

app = Flask(__name__)

# analyzer = OptimizedSentimentAnalyzer.load_model('optimized_sentiment_model_2_10000.pkl')
analyzer = OptimizedSentimentAnalyzer.load_model('optimized_sentiment_model_2_15000_11_Nov_2024_18hs_01min.pkl')
vocab = analyzer.vectorizer.vocabulary_

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
            translated_text = 'Lo siento, tengo problemas para traducir este texto.'
        
        # Get Gemini analysis
      #   analyzer_gemini = '50% positive, 50% negative'  # Replace with actual gemini.chat(text)
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
        else:
            none_gemini = 'None'
            positive_gemini = 0
            negative_gemini = 0
        
        # Prepare response data
        if filtered_text:
            result = analyzer.predict(filtered_text)
            print('result:', result['probabilities']['positive'])
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
                'none_gemini': none_gemini
            }
        else:
            print('filtered_text:', filtered_text)
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