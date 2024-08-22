# Avihai Mordechay 318341278, Omer Zamir 208552620
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def process_data():
    """
    Loads, preprocesses, and normalizes the dataset.

    Returns:
        X (numpy.ndarray): Normalized features.
        y (pandas.Series): Target labels.
    """

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


# Part 1:
def cross_validation(X, y):
    """
     Performs 5-fold cross-validation with logistic regression.

     Returns:
         float: Average accuracy score.
     """
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross Validation Scores (K-Fold):", scores)
    # Calculate average accuracy
    avg_accuracy = np.mean(scores)
    return avg_accuracy


# Part 2:
def random_split(X, y):
    """
       Evaluates logistic regression with 5 random 50% train/test splits.

       Returns:
           float: Average accuracy score.
       """
    accuracies = []

    for _ in range(5):  # Repeat 5 times
        # Randomly split the data into train and test (50% each)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

        # Initialize and train the logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    print("Random Split Accuracies:", accuracies)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


# Part 3:
def adaboost(X, y, n_estimators=5):
    """
    Trains an AdaBoost ensemble of logistic regression models.

    Args:
        n_estimators (int): Number of base models.

    Returns:
        float: Accuracy score.
    """
    # Replace 0 with -1 in y
    y = np.where(y == 0, -1, y)

    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    models = []
    alphas = []
    for _ in range(n_estimators):
        # Initialize a logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y, sample_weight=weights)
        predictions = model.predict(X)

        # Calculate the error
        error = np.sum(weights * (predictions != y)) / np.sum(weights)

        # Compute alpha
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        alphas.append(alpha)
        models.append(model)

        # Update the weights
        weights *= np.exp(alpha * (predictions != y))
        weights /= np.sum(weights)

    # Make predictions with the final model
    final_predictions = np.zeros(n_samples)
    for model, alpha in zip(models, alphas):
        final_predictions += alpha * model.predict(X)

    # Sign function to convert predictions to binary labels
    final_predictions = np.sign(final_predictions)

    # Calculate accuracy
    accuracy = np.mean(final_predictions == y)
    return accuracy


if __name__ == "__main__":
    # Process the data
    X, y = process_data()

    # Perform Cross Validation
    avg_accuracy_cv = cross_validation(X, y)
    print(f"Average accuracy from Cross Validation (K-Fold): {avg_accuracy_cv:.5f}")

    # Perform Random Split
    avg_accuracy_random_split = random_split(X, y)
    print(f"Average accuracy from Random 50% Split: {avg_accuracy_random_split:.5f}")

    # Perform AdaBoost with Logistic Regression
    ada_accuracy = adaboost(X, y, n_estimators=5)
    print(f"Accuracy from AdaBoost with Logistic Regression: {ada_accuracy:.5f}")
