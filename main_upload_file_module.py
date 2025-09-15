"""
This module implements a Flask web application for document classification.

It provides a web interface to upload a document (PDF, DOCX, or TXT),
and then uses a pre-trained machine learning model to predict the document's
category.
"""

from flask import Flask, request, render_template
import joblib
import PyPDF2
from docx import Document
import numpy as np

app = Flask(__name__)

def _extract_text_from_pdf(file):
    """Extracts text from a PDF file.

    Args:
        file: A file-like object representing the PDF file.

    Returns:
        The extracted text from the PDF file as a single string.

    Raises:
        PyPDF2.errors.PdfReadError: If the PDF file is invalid or cannot be read.
    """
    document_text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        document_text += page.extract_text() + "\n"
    return document_text

def _extract_text_from_docx(file):
    """Extracts text from a DOCX file.

    Args:
        file: A file-like object representing the DOCX file.

    Returns:
        The extracted text from the DOCX file as a single string.

    Raises:
        Exception: If the DOCX file is invalid or cannot be read.
    """
    document_text = ""
    doc = Document(file)
    document_text = "\n".join([para.text for para in doc.paragraphs])
    return document_text

def _extract_text_from_txt(file):
    """Extracts text from a TXT file.

    Note:
        This function assumes the file is encoded in 'ISO-8859-1'.
        This may need to be adjusted for other file encodings.

    Args:
        file: A file-like object representing the TXT file.

    Returns:
        The extracted text from the TXT file as a single string.
    """
    return file.read().decode('ISO-8859-1')

@app.route('/')
def upload_form():
    """Renders the file upload form.

    Returns:
        A rendered HTML template containing the upload form.
    """
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, extracts text, and predicts the category.

    This function processes uploaded files, extracts text content based on the
    file type (PDF, DOCX, or TXT), and then uses a loaded scikit-learn model
    to predict the document's category.

    Returns:
        A string indicating the predicted category and confidence score,
        or an error message if the file is invalid or not supported.
    """
    model = joblib.load('trained_model.pkl')
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    try:
        if file.filename.endswith('.pdf'):
            document_text = _extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            document_text = _extract_text_from_docx(file)
        elif file.filename.endswith('.txt'):
            document_text = _extract_text_from_txt(file)
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error processing file: {e}"

    # Make predictions using the model
    prediction = model.predict([document_text])
    confidence = model.predict_proba([document_text])

    return f"Predicted Category: {prediction[0]}, Confidence: {np.max(confidence)}"

if __name__ == '__main__':
    app.run(debug=True)
