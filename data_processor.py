import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from utils import stem_tokenizer, collect_file_paths_and_labels
from model_trainer import train_model

def prepare_and_train_model():
    """
    Prepare the data from the spam, easy_ham, and hard_ham folders, and train the Naive Bayes model.

    Returns:
        model: Trained Naive Bayes model.
        accuracy: Accuracy of the model on the test set.
    """
    base_path = os.path.dirname(__file__)

    # Collect file paths and labels
    file_paths, labels = collect_file_paths_and_labels(base_path)

    # Vectorize the text data
    vectorizer = CountVectorizer(
        tokenizer=stem_tokenizer,  # Use the custom tokenizer with stemming
        stop_words='english',      # Remove common stop words
        max_features=10000,        # Limit the vocabulary size to the top 10,000 features
        ngram_range=(1, 2)         # Include both unigrams and bigrams for richer context
    )

    all_texts = []
    for file in file_paths:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:  # Ignore encoding errors
            all_texts.append(f.read())

    X = vectorizer.fit_transform(all_texts)
    y = labels

    # Train the model
    model, accuracy = train_model(X, y)

    print(f"Model trained with accuracy: {accuracy:.2f}")
    return model, accuracy

