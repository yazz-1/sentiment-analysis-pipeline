import pytest
import sys
import os

# Agregar el directorio src al path para los imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_reviews():
    """Fixture que proporciona reviews de ejemplo a todos los tests"""
    return {
        'positive': "This movie is absolutely fantastic! Great acting and story.",
        'negative': "Terrible movie, complete waste of time. Poor acting.",
        'neutral': "The movie was okay, nothing special."
    }

@pytest.fixture
def sample_dataframe():
    """Fixture que proporciona DataFrame de prueba"""
    import pandas as pd
    return pd.DataFrame({
        'text': ['Great movie!', 'Terrible film', 'It was okay'],
        'label': [1, 0, 1]
    })
