from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, X_test, y_test):
    """
    Train a Naive Bayes model, evaluate it on the test set, and perform cross-validation.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Testing feature matrix.
        y_test: Testing labels.

    Returns:
        model: Trained Naive Bayes model.
        accuracy: Accuracy of the model on the test set.
        cv_accuracy: Cross-validated accuracy of the model.
    """
    model = MultinomialNB()

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    accuracy = model.score(X_test, y_test)

    return model, accuracy, cv_accuracy
