import pytest
from main_upload_file_module import app
from unittest.mock import patch, MagicMock
import io
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_form(client):
    """Test the upload form page."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"<h1>Upload new File</h1>" in rv.data

@patch('main_upload_file_module.joblib.load')
def test_upload_txt_file(mock_joblib_load, client):
    """Test uploading a txt file."""
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
    """Test uploading a pdf file."""
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
    """Test uploading a docx file."""
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
    """Test uploading an unsupported file type."""
    data = {
        'file': (io.BytesIO(b"this is a test"), 'test.zip')
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"Unsupported file type" in rv.data

def test_upload_no_file(client):
    """Test uploading with no file."""
    data = {}
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"No file part" in rv.data

def test_upload_empty_filename(client):
    """Test uploading with an empty filename."""
    data = {
        'file': (io.BytesIO(b"this is a test"), '')
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    assert b"No selected file" in rv.data
