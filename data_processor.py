import re
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from utils import stem_tokenizer, collect_file_paths_and_labels
from model_trainer import train_model
from joblib import dump, load

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_and_train_model(base_path=None):
    """
    Prepare the data from the spam, easy_ham, and hard_ham folders, and train the Naive Bayes model.

    The expected folder structure is:
    ├── spam
    │   ├── spam1.txt
    │   ├── spam2.txt
    │   └── ...
    ├── easy_ham
    │   ├── easy_ham1.txt
    │   ├── easy_ham2.txt
    │   └── ...
    └── hard_ham
        ├── hard_ham1.txt
        ├── hard_ham2.txt
        └── ...

    Args:
        base_path (str, optional): Base directory containing the folders. Defaults to the directory of this script.

    Returns:
        model: Trained Naive Bayes model.
        accuracy: Accuracy of the model on the test set.
    """
    if base_path is None:
        base_path = os.path.dirname(__file__)  # Default to the directory of this script

    try:
        # Collect file paths and labels
        file_paths, labels = collect_file_paths_and_labels(base_path)

        # Vectorize the text data
        vectorizer = CountVectorizer(
            tokenizer=stem_tokenizer,   # Custom tokenizer handles tokenization and stemming
            stop_words='english',       # Remove common stop words
            max_features=10000,         # Limit vocabulary size
            ngram_range=(1, 2),         # Unigrams and bigrams
        )

        all_texts = []
        for file in file_paths:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:  # Ignore encoding errors
                    all_texts.append(f.read())
            except Exception as e:
                logging.error(f"Error reading file {file}: {e}")

        X = vectorizer.fit_transform(all_texts)
        y = labels

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model, accuracy, cvAccuracy = train_model(X_train, y_train, X_test, y_test)

        logging.info(f"Model trained successfully! Test Accuracy: {accuracy:.2f}, Cross-Validation Accuracy: {cvAccuracy:.2f}")
        return model, accuracy, cvAccuracy

    except Exception as e:
        logging.error(f"An error occurred during model preparation and training: {e}")
        raise

def save_model_and_vectorizer(model, vectorizer, model_path="model.joblib", vectorizer_path="vectorizer.joblib"):
    """Save the trained model and vectorizer to disk."""
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    logging.info(f"Model saved to {model_path} and vectorizer saved to {vectorizer_path}.")

def load_model_and_vectorizer(model_path="model.joblib", vectorizer_path="vectorizer.joblib"):
    """Load the trained model and vectorizer from disk."""
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    logging.info(f"Model loaded from {model_path} and vectorizer loaded from {vectorizer_path}.")
    return model, vectorizer

def classify_email(email_text, model, vectorizer):
    """Classify a new email using the trained model and vectorizer."""
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)
    return prediction[0]

