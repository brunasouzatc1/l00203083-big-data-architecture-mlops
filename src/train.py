
import os

import pickle



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report



from preprocess import load_and_preprocess





MODEL_PATH = "model.pkl"





def train_model():

    X_train, X_test, y_train, y_test = load_and_preprocess()



    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )



    model.fit(X_train, y_train)



    predictions = model.predict(X_test)



    accuracy = accuracy_score(y_test, predictions)



    print("Model training completed!")

    print(f"Test accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(classification_report(y_test, predictions))



    with open(MODEL_PATH, "wb") as file:

        pickle.dump(model, file)



    print(f"\nModel saved to: {MODEL_PATH}")





if __name__ == "__main__":

    train_model()

