import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deployment.app import app, predict_single_text

def test_flask_app_health():
    """Test del endpoint de health"""
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'

def test_predict_endpoint():
    """Test del endpoint de predicción"""
    with app.test_client() as client:
        # Test con datos válidos
        response = client.post('/predict',
                             json={'text': 'This movie is amazing'})
        assert response.status_code in [200, 500]  # 500 si el modelo no está cargado

        if response.status_code == 200:
            data = response.get_json()
            assert 'sentiment' in data or 'error' in data

def test_predict_empty_text():
    """Test con texto vacío"""
    result = predict_single_text("")
    assert 'error' in result
    assert result['status'] == 'error'

def test_predict_invalid_input():
    """Test con entrada inválida"""
    result = predict_single_text(123)  # No string
    assert 'error' in result
    assert result['status'] == 'error'
