import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def get_clean_data():
    data = pd.read_csv('./data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, "B": 0})

    print(data.head())
    return data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, shuffle=True,
                                                        random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f'Accuracy of the Model: {accuracy_score(y_test, y_pred)}')
    print(f'Classification report: \n{classification_report(y_test, y_pred)}')
    return model, scalar



def main():
    data = get_clean_data()

    # Create the model
    model, scalar = create_model(data)

    # Train the model
    # train(model)

    # Test the model
    # test_model(model)

    # Evaluate the model
    # evaluate(model)
if __name__== '__main__':
    main()