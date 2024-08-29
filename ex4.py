import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def process_data():
    """
    Loads and preprocesses the dataset:
    - Reads the CSV file and handles missing values.
    - Encodes categorical variables to numeric values.
    - Normalizes the feature values.

    Returns:
        X (numpy.ndarray): Normalized feature matrix.
        y (pandas.Series): Target labels.
    """
    # Load the data with column names and handle missing values
    df = pd.read_csv('adult.data', header=None, na_values=' ?',
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                            'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    # Fill missing values with the last known value
    df.fillna(method='ffill', inplace=True)

    # Convert categorical variables to numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split data into features and target variable
    X = df.drop('income', axis=1)
    y = df['income']

    # Normalize feature values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def cross_validation(X, y):
    """
    Evaluates the decision tree classifier using 5-fold cross-validation.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (pandas.Series): Target labels.

    Returns:
        float: Average accuracy score from cross-validation.
    """
    # Initialize the decision tree classifier
    model = DecisionTreeClassifier()

    # Perform k-fold cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross Validation Scores (K-Fold):", scores)
    
    # Calculate and return average accuracy
    avg_accuracy = np.mean(scores)

    plot_accuracy(scores, 'Cross-Validation Accuracy Scores')

    return avg_accuracy


def random_split(X, y):
    """
    Evaluates the decision tree classifier using 5 random 50% train/test splits.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (pandas.Series): Target labels.

    Returns:
        float: Average accuracy score from random splits.
    """
    accuracies = []

    for _ in range(5):  # Repeat 5 times
        # Randomly split data into train and test sets (50% each)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

        # Train and evaluate the decision tree classifier
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate and store accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plot_accuracy(np.array(accuracies), 'Random (50%) Accuracy Scores')

    print("Random Split Accuracies:", accuracies)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


def adaboost(X, y, n_estimators=50, max_depth=5):
    """
    Trains an AdaBoost ensemble of decision tree classifiers.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (pandas.Series): Target labels.
        n_estimators (int): Number of base models (default is 50).
        max_depth (int): Maximum depth of the decision trees (default is 5).

    Returns:
        float: Accuracy score of the AdaBoost model.
    """
    # Convert target labels from 0 to -1 for AdaBoost
    y = np.where(y == 0, -1, y)

    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    models = []
    alphas = []
    
    for i in range(n_estimators):
        # Train a weak decision tree classifier
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X, y, sample_weight=weights)
        predictions = model.predict(X)

        # Compute error and alpha
        error = np.sum(weights * (predictions != y)) / np.sum(weights)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        alphas.append(alpha)
        models.append(model)

        # Update sample weights
        weights *= np.exp(alpha * (predictions != y))
        weights /= np.sum(weights)

    # Aggregate predictions from all models
    final_predictions = np.zeros(n_samples)
    for model, alpha in zip(models, alphas):
        final_predictions += alpha * model.predict(X)

    # Convert predictions to binary labels
    final_predictions = np.sign(final_predictions)

    # Calculate accuracy of the final ensemble model
    accuracy = np.mean(final_predictions == y)
    
    return accuracy


def plot_decision_boundary_with_pca(X, y):
    """
    Reduces features to 2 dimensions using PCA and plots the data points and decision boundary.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
    """
    # Apply PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Train a decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_pca, y)

    # Calculate accuracy
    accuracy = accuracy_score(y, model.predict(X_pca))

    plt.figure(figsize=(10, 6))

    # Plot the data points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('32561 people in the United States\nReducing 14 features to 2 features by using PCA')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Under 50K', 'Over 50K'])

    plt.show()


def plot_accuracy(scores, title):
    """
    Plots accuracy scores from cross-validation or random splits.

    Args:
        scores (numpy.ndarray): Array of accuracy scores.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 5))

    # Plot the accuracy scores
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')

    # Add text annotations
    for i, score in enumerate(scores):
        plt.text(i + 1, score, f'{score:.5f}', ha='center', va='bottom')

    plt.xlabel('Fold/Iteration')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0.8, 0.9)  # Adjust limits to focus on relevant values
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Process and prepare the data
    X, y = process_data()

    # Plot decision boundary with PCA
    plot_decision_boundary_with_pca(X, y)

    # Perform Cross-Validation
    avg_accuracy_cv = cross_validation(X, y)
    print(f"Average accuracy from Cross Validation (K-Fold): {avg_accuracy_cv:.5f}")

    # Perform Random Split
    avg_accuracy_random_split = random_split(X, y)
    print(f"Average accuracy from Random 50% Split: {avg_accuracy_random_split:.5f}")

    # Perform AdaBoost with Decision Tree
    ada_accuracy = adaboost(X, y, n_estimators=200)
    print(f"Accuracy from AdaBoost with Decision Tree: {ada_accuracy:.5f}")
