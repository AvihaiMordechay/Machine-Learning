import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def process_data():
    # Load the data
    df = pd.read_csv('adult.data', header=None, na_values=' ?',
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                            'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    # Fill the empty cells with the last known cell
    df.fillna(method='ffill', inplace=True)

    # Convert categorical variables to numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split the data to X and Y
    X = df.drop('income', axis=1)
    y = df['income']

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def logistic_regression_cv(X, y):
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(scores)
    # Calculate average accuracy
    avg_accuracy = np.mean(scores)
    return avg_accuracy


if __name__ == "__main__":
    # Process the data
    X, y = process_data()

    # Perform Cross Validation
    avg_accuracy = logistic_regression_cv(X, y)
    print(f"Average accuracy from Cross Validation: {avg_accuracy:.4f}")
