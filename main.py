import pandas as pd
import numpy as np

def get_clean_data():
    data = pd.read_csv('./data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, "B": 0})

    print(data.head())
    return data

def main():
    data = get_clean_data()

    # Create the model
    # model = create_model()

    # Train the model
    # train(model)

    # Evaluate the model
    # evaluate(model)
if __name__== '__main__':
    main()