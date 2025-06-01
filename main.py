import os
from data_processor import prepare_and_train_model, save_model_and_vectorizer, load_model_and_vectorizer, classify_email

if __name__ == "__main__":
    model_path = "model.joblib"
    vectorizer_path = "vectorizer.joblib"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        # Load the model and vectorizer if they exist
        model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        print("Loaded existing model and vectorizer.")
    else:
        # Trigger the entire flow to prepare and train the model
        model, accuracy, cvAccuracy = prepare_and_train_model()
        save_model_and_vectorizer(model, model.vectorizer, model_path, vectorizer_path)
        print("Trained and saved the model and vectorizer.")

    # Classify a new email
    new_email = "Congratulations! You've won a free ticket to Bahamas."
    prediction = classify_email(new_email, model, vectorizer)
    print(f"The email was classified as: {'Spam' if prediction == 1 else 'Not Spam'}")

    print("Flow completed successfully.")
