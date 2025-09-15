from flask import Flask, request, render_template
import joblib
import PyPDF2
from docx import Document
import numpy as np

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    model = joblib.load('trained_model.pkl')
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Initialize an empty string to hold the document text
    document_text = ""

    # Check the file extension and read the content accordingly
    if file.filename.endswith('.pdf'):
        # Handle PDF files
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                document_text += page.extract_text() + "\n"
        except PyPDF2.errors.PdfReadError:
            return "Invalid PDF file"
    elif file.filename.endswith('.docx'):
        # Handle Word documents
        try:
            doc = Document(file)
            document_text = "\n".join([para.text for para in doc.paragraphs])
        except:
            return "Invalid DOCX file"
    elif file.filename.endswith('.txt'):
        # Handle plain text files
        document_text = file.read().decode('ISO-8859-1')  # Adjust encoding as needed
    else:
        return "Unsupported file type"

    # Make predictions using the model
    prediction = model.predict([document_text])
    confidence = model.predict_proba([document_text])

    return f"Predicted Category: {prediction[0]}, Confidence: {np.max(confidence)}"

if __name__ == '__main__':
    app.run(debug=True)
