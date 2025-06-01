import os
import re
from nltk.stem import PorterStemmer

def stem_tokenizer(text):
    """
    Tokenize and stem the input text.

    Args:
        text: Input string.

    Returns:
        List of stemmed tokens.
    """
    stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens]

def collect_file_paths_and_labels(base_path):
    """
    Collect file paths and labels from the spam, easy_ham, and hard_ham folders.

    Args:
        base_path: Base directory containing the folders.

    Returns:
        file_paths: List of file paths.
        labels: List of corresponding labels.
    """
    spam_folder = os.path.join(base_path, 'spam')
    easy_ham_folder = os.path.join(base_path, 'easy_ham')
    hard_ham_folder = os.path.join(base_path, 'hard_ham')

    file_paths = []
    labels = []

    for file_name in os.listdir(spam_folder):
        file_paths.append(os.path.join(spam_folder, file_name))
        labels.append(1)

    for file_name in os.listdir(easy_ham_folder):
        file_paths.append(os.path.join(easy_ham_folder, file_name))
        labels.append(0)

    for file_name in os.listdir(hard_ham_folder):
        file_paths.append(os.path.join(hard_ham_folder, file_name))
        labels.append(0)

    return file_paths, labels
