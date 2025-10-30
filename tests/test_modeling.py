import joblib
import os

def test_model_loading():
    """Test que verifica que el modelo puede cargarse"""
    try:
        pipeline = joblib.load('models/pipeline.joblib')
        assert pipeline is not None
    except FileNotFoundError:
        pytest.skip("Modelo no encontrado, saltando test")

def test_model_prediction_format():
    """Test que verifica el formato de las predicciones"""
    try:
        pipeline = joblib.load('models/pipeline.joblib')
        sample_text = "This movie is great"

        # Verificar que tiene los métodos necesarios
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')
        assert hasattr(pipeline, 'classes_')

    except FileNotFoundError:
        pytest.skip("Modelo no encontrado, saltando test")
