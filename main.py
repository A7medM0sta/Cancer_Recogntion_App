import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pickle5 as pickle
import os


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
    report = classification_report(y_test, y_pred, output_dict=True)
    # Generate confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    # Convert report to DataFrame for easier manipulation
    df_report = pd.DataFrame(report).transpose()

    # Prepare data for plotting
    categories = df_report.index[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = df_report['precision'][:-3]
    recall = df_report['recall'][:-3]
    f1_score = df_report['f1-score'][:-3]

    # Create a subplot figure with 2 rows
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Classification Report', 'Confusion Matrix'),
                        vertical_spacing=0.15)

    # Add the bar chart to the first row
    fig.add_trace(go.Bar(name='Precision', x=categories, y=precision, marker_color='SkyBlue'), row=1, col=1)
    fig.add_trace(go.Bar(name='Recall', x=categories, y=recall, marker_color='IndianRed'), row=1, col=1)
    fig.add_trace(go.Bar(name='F1-Score', x=categories, y=f1_score, marker_color='LightGreen'), row=1, col=1)

    # Add the heatmap to the second row
    heatmap = ff.create_annotated_heatmap(z=matrix, x=["Predicted Benign", "Predicted Malignant"],
                                          y=["Actual Benign", "Actual Malignant"], colorscale="Viridis")
    for i in range(len(heatmap.data)):
        fig.add_trace(heatmap.data[i], row=2, col=1)

    # Customize layout
    fig.update_layout(title_text="Classification Report and Confusion Matrix", barmode='group')
    fig.update_layout(height=800, showlegend=True)

    # Show the figure
    fig.show()
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
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(model_dir, 'scalar.pkl'), "wb") as f:
        pickle.dump(scalar, f)



if __name__== '__main__':
    main()