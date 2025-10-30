import spacy
from src.preprocessing_02 import clean_doc
import pytest

def test_clean_text_basic():
    """Test básico de limpieza - USANDO MODELO REAL"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("Modelo spaCy no disponible")

    doc = nlp("I LOVED this movie!!! <br> Visit http://example.com")
    cleaned = clean_doc(doc)

    # Verificaciones más realistas
    assert "love" in cleaned  # "LOVED" debería lematizarse a "love"
    assert "movie" in cleaned
    # Las stopwords "I", "this" deberían eliminarse
    assert "i" not in cleaned
    assert "this" not in cleaned

def test_clean_doc_stopwords_removal():
    """Test de eliminación de stopwords - CORREGIDO"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("Modelo spaCy no disponible")

    # Texto que solo contiene stopwords conocidas
    doc = nlp("the and or but this that")
    cleaned = clean_doc(doc)

    # Con modelo real, todas deberían eliminarse
    assert len(cleaned) == 0

def test_clean_doc_empty_text():
    """Test con texto vacío"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("Modelo spaCy no disponible")

    doc = nlp("")
    cleaned = clean_doc(doc)
    assert cleaned == []

def test_clean_doc_special_characters():
    """Test con caracteres especiales"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("Modelo spaCy no disponible")

    doc = nlp("Hello! @user123 #tag 123numbers")
    cleaned = clean_doc(doc)
    cleaned_text = " ".join(cleaned)

    # Verificar que no hay caracteres especiales
    assert "@" not in cleaned_text
    assert "#" not in cleaned_text
    assert "123" not in cleaned_text

def test_clean_doc_lemmatization():
    """Test de lematización"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("Modelo spaCy no disponible")

    doc = nlp("running runs ran beautiful beautifully")
    cleaned = clean_doc(doc)

    # En modelo real, deberían lematizarse
    assert "run" in cleaned  # "running", "runs", "ran" → "run"
    assert "beautiful" in cleaned  # "beautifully" → "beautiful"
