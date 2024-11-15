# Sentiment Analysis Web Application

![alt text](https://i.imgur.com/JeSH8qd.gif)

## Análisis basado en: fine-food-reviews de amazon

Esta es una aplicación web de análisis de sentimiento desarrollada en HTML, CSS (con Tailwind CSS), JavaScript y Python (Django en el backend). La aplicación permite a los usuarios ingresar un texto y recibir un análisis de sentimiento que muestra las puntuaciones de emociones positivas y negativas.

Este proyecto implementa un modelo de análisis de sentimiento en Python utilizando `TfidfVectorizer` y `SGDClassifier`, optimizado para reducir recursos de procesamiento y memoria. El modelo permite realizar predicciones de sentimientos en texto, categorizando las opiniones como "Positivas" o "Negativas" con un cálculo de confianza.
Además implementa API gemini para analizar el sentimiento de la frase y contrastar el resultado del modelo!

## APP: link

https://analyzerreviewfood-production.up.railway.app/

## Deploy en:

https://railway.app/

## Características

- **Análisis de Sentimientos**: Los usuarios pueden ingresar un texto para analizar y recibir puntuaciones de sentimientos positivos y negativos.
- **Visualización**: Los resultados se presentan en un gráfico de barras horizontal utilizando Plotly.
- **Interfaz Moderna**: Estilos de interfaz con Tailwind CSS para un diseño limpio y atractivo.
- **Manejo de Casos Sin Contenido**: Si el texto ingresado no contiene frases significativas, se muestra un mensaje específico.


## Descripción del Proyecto

El modelo de análisis de sentimiento utiliza técnicas avanzadas de procesamiento de texto para convertir opiniones en datos estructurados. Se apoya en el uso de `TfidfVectorizer` para vectorizar texto y `SGDClassifier` como clasificador, ofreciendo una solución eficiente y escalable para grandes volúmenes de datos.

### Características Clave

- **Preprocesamiento Optimizado:** 
  - Conversión de texto a minúsculas, eliminación de URLs, puntuación, y aplicación de stemming para reducir la dimensionalidad de las características.
  - Detección y manejo de palabras de negación para mejorar la precisión del modelo.

- **Vectorización y Clasificación:**
  - `TfidfVectorizer` con un máximo de 10000 características y bigramas.
  - Clasificación mediante `SGDClassifier`, ideal para grandes volúmenes de texto.

- **Entrenamiento y Evaluación:**
  - Entrena el modelo usando una muestra de los datos y evalúa la precisión en el conjunto de prueba.

## Requisitos

- **Python 3.8+**
- django
- numpy
- pandas
- scikit-learn
- nltk
- googletrans==4.0.0rc1
- gunicorn
- dj-database-url
- psycopg2-binary
- python-dotenv

Para instalar los requisitos:
```bash
pip install -r requirements.txt
```

## Instrucciones de Uso

1. **Inicializar el Analizador:** 
   - Importar `SentimentAnalyzer` y crear una instancia:
     ```python
     analyzer = SentimentAnalyzer()
     ```

2. **Entrenar el Modelo:**
   - Proveer el archivo de datos (e.g., `Reviews.csv`) con las columnas `Score` y `Text`.
   - Llamar al método `train` para entrenar el modelo, pasando el archivo CSV y el tamaño de la muestra:
     ```python
     analyzer.train('Reviews.csv', sample_size=50000)
     ```
     **Descargar DDBB:**
      - https://www.kaggle.com/code/chirag9073/amazon-fine-food-reviews-sentiment-analysis/notebook


3. **Guardar el Modelo:**
   - Guardar el modelo entrenado en un archivo `.pkl` para uso futuro:
     ```python
     analyzer.save_model('optimized_sentiment_model.pkl')
     ```

4. **Hacer Predicciones:**
   - Para realizar predicciones, llamar al método `predict` pasando el texto de entrada:
     ```python
     result = analyzer.predict("this product is amazing")
     print(result)
     ```

   - El resultado incluirá la predicción de sentimiento ("Positive" o "Negative") y un nivel de confianza.

5. **Cargar un Modelo Guardado:**
   - Para cargar un modelo previamente guardado, utilizar el método `load_model`:
     ```python
     analyzer = OptimizedSentimentAnalyzer.load_model('optimized_sentiment_model.pkl')
     ```

## Ejemplo de Uso en Línea de Comandos

```bash
python sentiment_analyzer.py
```

Este comando ejecutará el entrenamiento y realizará predicciones de prueba en algunos textos de ejemplo.

## Configuraciones

Algunas configuraciones ajustables incluyen:
- `ngram` y `features` para definir el rango de `n-grams` y el máximo de características.
- `sample_size` en el entrenamiento para controlar la cantidad de datos procesados.

## Créditos

Este modelo fue desarrollado utilizando `scikit-learn` y `nltk`, y está optimizado para aplicaciones que requieren una clasificación de sentimiento eficiente y ligera.

## DDBB y fuente
- https://www.kaggle.com/code/chirag9073/amazon-fine-food-reviews-sentiment-analysis/notebook

