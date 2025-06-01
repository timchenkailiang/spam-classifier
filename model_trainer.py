from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

def train_model(X, y):
    """
    Train a Naive Bayes model using cross-validation.

    Args:
        X: Feature matrix.
        y: Labels.

    Returns:
        model: Trained Naive Bayes model.
        accuracy: Cross-validated accuracy of the model.
    """
    model = MultinomialNB()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()
    model.fit(X, y)  # Train on the entire dataset
    return model, accuracy
