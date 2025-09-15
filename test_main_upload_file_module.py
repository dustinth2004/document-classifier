"""
This module contains tests for the Flask web application in main_upload_file_module.py.

It uses pytest and unittest.mock to test the application's routes and file handling logic.
"""

import pytest
from main_upload_file_module import app
from unittest.mock import patch, MagicMock
import io
import numpy as np

@pytest.fixture
def client():
    """A pytest fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_form(client):
    """Test that the upload form page loads correctly.

    Args:
        client: The Flask test client.
    """
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"<h1>Upload new File</h1>" in rv.data

@patch('main_upload_file_module.joblib.load')
def test_upload_txt_file(mock_joblib_load, client):
    """Test uploading a .txt file.

    This test mocks the model loading and checks if the application correctly
    processes a .txt file and returns the expected prediction.

    Args:
        mock_joblib_load: A mock for the joblib.load function.
        client: The Flask test client.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ['text']
    mock_model.predict_proba.return_value = np.array([[0.9]])
    mock_joblib_load.return_value = mock_model

    data = {
        'file': (io.BytesIO(b"this is a test txt file"), 'test.txt')
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"Predicted Category: text, Confidence: 0.9" in rv.data

@patch('main_upload_file_module.joblib.load')
def test_upload_pdf_file(mock_joblib_load, client):
    """Test uploading a .pdf file.

    This test mocks the model loading and the PDF reader to check if the
    application correctly processes a .pdf file and returns the expected prediction.

    Args:
        mock_joblib_load: A mock for the joblib.load function.
        client: The Flask test client.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ['pdf']
    mock_model.predict_proba.return_value = np.array([[0.8]])
    mock_joblib_load.return_value = mock_model

    with patch('main_upload_file_module.PyPDF2.PdfReader') as mock_pdf_reader:
        mock_reader_instance = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "this is a test pdf file"
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance

        data = {
            'file': (io.BytesIO(b"%PDF-1.5..."), 'test.pdf')
        }
        rv = client.post('/upload', data=data, content_type='multipart/form-data')
        assert rv.status_code == 200
        assert b"Predicted Category: pdf, Confidence: 0.8" in rv.data

@patch('main_upload_file_module.joblib.load')
def test_upload_docx_file(mock_joblib_load, client):
    """Test uploading a .docx file.

    This test mocks the model loading and the DOCX document parser to check
    if the application correctly processes a .docx file and returns the expected
    prediction.

    Args:
        mock_joblib_load: A mock for the joblib.load function.
        client: The Flask test client.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ['docx']
    mock_model.predict_proba.return_value = np.array([[0.7]])
    mock_joblib_load.return_value = mock_model

    with patch('main_upload_file_module.Document') as mock_docx:
        mock_doc_instance = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "this is a test docx file"
        mock_doc_instance.paragraphs = [mock_para]
        mock_docx.return_value = mock_doc_instance

        data = {
            'file': (io.BytesIO(b"word"), 'test.docx')
        }
        rv = client.post('/upload', data=data, content_type='multipart/form-data')
        assert rv.status_code == 200
        assert b"Predicted Category: docx, Confidence: 0.7" in rv.data

def test_upload_unsupported_file(client):
    """Test uploading an unsupported file type.

    This test checks if the application returns an 'Unsupported file type'
    error when a file with an unsupported extension is uploaded.

    Args:
        client: The Flask test client.
    """
    data = {
        'file': (io.BytesIO(b"this is a test"), 'test.zip')
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"Unsupported file type" in rv.data

def test_upload_no_file(client):
    """Test uploading with no file selected.

    This test checks if the application returns a 'No file part' error when
    the form is submitted without a file.

    Args:
        client: The Flask test client.
    """
    data = {}
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"No file part" in rv.data

def test_upload_empty_filename(client):
    """Test uploading a file with an empty filename.

    This test checks if the application returns a 'No selected file' error
    when a file with an empty filename is submitted.

    Args:
        client: The Flask test client.
    """
    data = {
        'file': (io.BytesIO(b"this is a test"), '')
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"No selected file" in rv.data
