# Document Classifier

## Description

This project is an AI-driven text classification application. It uses a machine learning model to categorize documents (PDF, DOCX, and TXT) based on their content. The application provides a simple web interface to upload documents and view the predicted category.

This project is intended as a demonstration of a simple document classification pipeline, including model training, a web interface, and testing.

## Tech Stack

*   **Programming Language:** Python
*   **Web Framework:** Flask
*   **Machine Learning:** scikit-learn, pandas, numpy
*   **File Handling:** PyPDF2, python-docx
*   **Testing:** pytest

## Project Structure

```
.
├── Model_Training.py
├── README.md
├── main_upload_file_module.py
├── requirements.txt
├── templates
│   └── upload.html
├── test_main_upload_file_module.py
├── train.csv
├── trained_model.pkl
└── ...
```

*   `main_upload_file_module.py`: The main Flask application file. It handles the web interface and the classification logic.
*   `Model_Training.py`: A script to train the document classification model.
*   `test_main_upload_file_module.py`: Pytest tests for the Flask application.
*   `test catagrise.py`: A script for advanced model evaluation and hyperparameter tuning.
*   `templates/upload.html`: The HTML template for the file upload form.
*   `requirements.txt`: A list of the Python dependencies for the project.
*   `train.csv`: The dataset used to train the model.
*   `trained_model.pkl`: The pre-trained model file.

## Getting Started

### Prerequisites

*   Python 3.6+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Web Application

To start the Flask web application, run the following command:

```bash
python main_upload_file_module.py
```

The application will be available at `http://127.0.0.1:5000`. You can use the web interface to upload a document and see the predicted category.

### Training the Model

To retrain the model, you can run the `Model_Training.py` script:

```bash
python Model_Training.py
```

This will train a new model on the `train.csv` dataset and save it as `trained_model.pkl`.

## Testing

To run the tests, use `pytest`:

```bash
pytest
```

The tests will verify the functionality of the Flask application, including file uploads and error handling.
