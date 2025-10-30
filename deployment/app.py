from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words

app = Flask(__name__)

pipeline = None
sentiment_mapping = None
models_loaded = False

def load_models():
    """Carga todos los modelos necesarios para la predicción"""
    global pipeline, sentiment_mapping, models_loaded

    try:
        # Cargar modelo entrenado
        pipeline = joblib.load('models/pipeline.joblib')
        print("✅ Modelo de sentiment analysis cargado")

        # Definir el mapeo de sentimientos
        sentiment_mapping = {'negative': 0, 'positive': 1}
        print("✅ Mapping de sentimientos definido")

        models_loaded = True

    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        models_loaded = False
        raise e

def ensure_models_loaded():
    """Verifica que los modelos estén cargados antes de procesar requests"""
    global models_loaded
    if not models_loaded:
        print("🔧 Cargando modelos bajo demanda...")
        load_models()

# Cargar modelo de spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
    print("✅ Modelo spaCy cargado para preprocesamiento")
except OSError:
    print("❌ No se pudo cargar spaCy, usando preprocesamiento simple")
    nlp = None

def clean_text(text):
    """
    Función de preprocesamiento CONSISTENTE con el entrenamiento
    """
    if not text or not isinstance(text, str):
        return ""

    # Si spaCy está disponible, usar el mismo procesamiento que en entrenamiento
    if nlp is not None:
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()  # Usar LEMMA no texto para consistencia
            for token in doc
            if token.is_alpha
            and token.lemma_.lower() not in stop_words
            and len(token.text) > 2
        ]
        return " ".join(tokens)
    else:
        # Fallback a preprocesamiento simple
        return simple_clean_text(text)

def simple_clean_text(text):
    """Preprocesamiento simple si spaCy no está disponible"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    # Stopwords básicas en inglés
    simple_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been'
    }
    words = [word for word in words if word not in simple_stopwords and len(word) > 2]
    return " ".join(words)

def preprocess_single_text(text: str) -> str:
    """Preprocesa un solo texto para predicción"""
    if nlp is None:
        return simple_clean_text(text)

    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]
    return " ".join(tokens)

def predict_single_text(text: str):
    """Predice el sentimiento de un texto individual"""
    try:
        if not text or not isinstance(text, str):
            return {"error": "Texto inválido", "status": "error"}

        # Verificar que el pipeline esté cargado
        if pipeline is None:
            return {"error": "Modelo no cargado", "status": "error"}

        cleaned_text = preprocess_single_text(text)

        # Realizar predicción
        prediction = pipeline.predict([cleaned_text])[0]
        probs = pipeline.predict_proba([cleaned_text])[0]

        # Obtener las probabilidades correctamente
        if hasattr(pipeline, 'classes_'):
            # Encontrar el índice de la clase positiva
            if 'pos' in pipeline.classes_:
                pos_index = list(pipeline.classes_).index("pos")
                prob_pos = float(probs[pos_index])
            else:
                # Si no hay 'pos', asumir que la última clase es positiva
                prob_pos = float(probs[-1])
        else:
            # Asumir estructura binaria estándar
            prob_pos = float(probs[1]) if len(probs) > 1 else float(probs[0])

        prob_neg = 1.0 - prob_pos

        # Determinar el resultado
        if prediction == "pos" or prediction == 1:
            result = "positive"
        elif prediction == "neg" or prediction == 0:
            result = "negative"
        else:
            result = str(prediction)

        return {
            "sentiment": result,
            "probability_pos": round(prob_pos * 100, 1),
            "probability_neg": round(prob_neg * 100, 1),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "error"}


# Rutas para la interfaz web

@app.route('/')
def home():
    """Página principal con el formulario de análisis de sentimientos"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Endpoint para la interfaz web que procesa el formulario
    y muestra los resultados en la página
    """
    try:
        # Obtener el texto del formulario
        review_text = request.form.get('review_text', '').strip()

        if not review_text:
            return render_template('index.html',
                                error="Por favor, escribe una reseña para analizar")

        # Realizar predicción
        result = predict_single_text(review_text)

        # Si hay error en la predicción
        if result.get('status') == 'error':
            return render_template('index.html',
                                error=f"Error analizando sentimiento: {result.get('error', 'Unknown error')}")

        # Determinar colores y emojis según el sentimiento
        sentiment = result.get('sentiment', 'neutral')
        if sentiment == 'positive':
            sentiment_color = 'success'
            sentiment_emoji = '😊'
            sentiment_text = 'POSITIVO'
        elif sentiment == 'negative':
            sentiment_color = 'danger'
            sentiment_emoji = '😞'
            sentiment_text = 'NEGATIVO'
        else:
            sentiment_color = 'warning'
            sentiment_emoji = '😐'
            sentiment_text = 'NEUTRO'

        # Renderizar resultado en la plantilla
        return render_template('index.html',
                            review_text=review_text,
                            result=result,
                            sentiment_color=sentiment_color,
                            sentiment_emoji=sentiment_emoji,
                            sentiment_text=sentiment_text)

    except Exception as e:
        return render_template('index.html',
                            error=f"Error procesando la reseña: {str(e)}")

# Rutas de la API

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({
        "status": "healthy",
        "message": "Sentiment Analysis API is running",
        "model_loaded": pipeline is not None,
        "models_loaded": models_loaded,
        "sentiment_mapping": sentiment_mapping
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint API para predecir sentimiento de texto(s).
    """
    try:
        # Recibir datos
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400

        # Verificar que los modelos están cargados
        ensure_models_loaded()

        # Procesar según el formato recibido
        if 'reviews' in data:
            # Múltiples textos
            texts = data['reviews']
            if not isinstance(texts, list):
                return jsonify({
                    "error": "'reviews' must be a list",
                    "status": "error"
                }), 400

            results = [predict_single_text(text) for text in texts]

            return jsonify({
                "predictions": results,
                "total_processed": len(results),
                "status": "success"
            })

        elif 'text' in data:
            # Un solo texto
            text = data['text']
            result = predict_single_text(text)

            return jsonify(result)

        else:
            return jsonify({
                "error": "Must provide either 'text' or 'reviews' in JSON",
                "status": "error"
            }), 400

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "status": "error"
        }), 500


@app.before_request
def before_each_request():
    """Se ejecuta antes de cada request"""
    global models_loaded
    if not models_loaded:
        print("🔄 Cargando modelos antes del primer request...")
        try:
            load_models()
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")

# Cargar modelos inmediatamente al importar
print("🚀 Inicializando Sentiment Analysis API con Interfaz Web...")

try:
    load_models()
    print("✅ Modelos cargados exitosamente durante la inicialización")
except Exception as e:
    print(f"⚠️  No se pudieron cargar los modelos durante la inicialización: {e}")
    print("📝 Los modelos se cargarán bajo demanda")

if __name__ == '__main__':
    print("🎯 Iniciando servidor Flask...")
    app.run(host='0.0.0.0', port=8000, debug=False)
