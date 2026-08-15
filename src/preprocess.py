
import pandas as pd

from sklearn.model_selection import train_test_split



INPUT_FILE = "data/l00203083_Iris_dataset.csv"



FEATURES = [

    "sepal_length",

    "sepal_width",

    "petal_length",

    "petal_width"

]



TARGET = "species"





def load_and_preprocess():

    df = pd.read_csv(INPUT_FILE)



    X = df[FEATURES]

    y = df[TARGET]



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    return X_train, X_test, y_train, y_test





if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_and_preprocess()



    print("Dataset loaded successfully!")

    print(f"Training samples: {len(X_train)}")

    print(f"Testing samples: {len(X_test)}")

    print(f"Features: {FEATURES}")

    print(f"Target: {TARGET}")

