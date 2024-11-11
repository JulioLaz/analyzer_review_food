from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from sentiment_analyzer import SentimentAnalyzer
from deep_translator import GoogleTranslator
# import langid
import gemini
import time

analyzer = SentimentAnalyzer.load_model('myapp/optimized_sentiment_model_2_10000.pkl')
vocab = analyzer.vectorizer.vocabulary_  # Obtiene el vocabulario del modelo cargado

@csrf_exempt
def analyze_sentiment(request):
    start_time = time.time()  # Iniciar el temporizador
    
    if request.method == 'POST':
        text = request.POST.get('text', '')
        try:
            translated_text = GoogleTranslator(source='auto', target='es').translate(text)
            print(f"Translation translated_text: {translated_text}")
        except Exception as e:
            translated_text = 'Lo siento, tengo problemas para traducir este texto'

        # analyzer_gemini = gemini.chat(text)
        analyzer_gemini= '50% positive, 50% negative'

        print('analyzer_gemini :', analyzer_gemini)

        filtered_text = " ".join(word for word in text.split() if word in vocab)
        print("Texto filtrado según vocabulario:", filtered_text)
        
        none_gemini = True
        if "positive" in analyzer_gemini or "negative" in analyzer_gemini:
            parts = analyzer_gemini.split(',')
            positive_gemini = int(parts[0].split('%')[0].strip())
            negative_gemini = int(parts[1].split('%')[0].strip())
        else:
            none_gemini = 'None'
            positive_gemini = 0
            negative_gemini = 0

        if filtered_text:
            result = analyzer.predict(filtered_text)
            print('result: ', result)
            response_data = {
                'filtered_text': filtered_text,
                'result': result,
                'positive_gemini': positive_gemini,
                'negative_gemini': negative_gemini,
                'translated_text': translated_text,
                'gemini_result': analyzer_gemini,
                'sentiment': 'unknown',
                'none_gemini': none_gemini
            }
        else:
            print('filtered_text: ', filtered_text)
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

        end_time = time.time()  # Finalizar el temporizador
        response_data['latency'] = round(end_time - start_time,3)  # Calcular la latencia
        print('Latencia: ', response_data)
        return JsonResponse(response_data)
    return render(request, 'sentiment_app/analyze.html')
