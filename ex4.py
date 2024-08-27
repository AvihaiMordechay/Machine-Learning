# Avihai Mordechay 318341278, Omer Zamir 208552620
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
    Performs 5-fold cross-validation with a decision tree classifier.

    Returns:
        float: Average accuracy score.
    """
    # Initialize the decision tree classifier
    model = DecisionTreeClassifier()

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross Validation Scores (K-Fold):", scores)
    
    # Calculate average accuracy
    avg_accuracy = np.mean(scores)

    plot_accuracy(scores, 'Cross-Validation Accuracy Scores')

    return avg_accuracy


# Part 2:
def random_split(X, y):
    """
    Evaluates decision tree classifier with 5 random 50% train/test splits.

    Returns:
        float: Average accuracy score.
    """
    accuracies = []

    for _ in range(5):  # Repeat 5 times
        # Randomly split the data into train and test (50% each)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

        # Initialize and train the decision tree classifier
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plot_accuracy(np.array(accuracies), 'Random (50%) Accuracy Scores')

    print("Random Split Accuracies:", accuracies)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


# part 3 
def adaboost(X, y, n_estimators=50, max_depth=5):
    """
    Trains an AdaBoost ensemble of decision tree classifiers with tunable max_depth.

    Args:
        n_estimators (int): Number of base models.
        max_depth (int): Maximum depth of the decision trees.

    Returns:
        float: Accuracy score.
    """
    # Replace 0 with -1 in y
    y = np.where(y == 0, -1, y)

    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    models = []
    alphas = []
    for i in range(n_estimators):
        # Use a weak decision tree classifier
        model = DecisionTreeClassifier(max_depth=7)
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

        # Calculate accuracy for the current model
        accuracy = np.mean(np.sign(predictions) == y)
    
    # Make predictions with the final model
    final_predictions = np.zeros(n_samples)
    for model, alpha in zip(models, alphas):
        final_predictions += alpha * model.predict(X)

    # Sign function to convert predictions to binary labels
    final_predictions = np.sign(final_predictions)

    # Calculate accuracy of the final ensemble model
    accuracy = np.mean(final_predictions == y)
    
    return accuracy


# Graphs:
def plot_decision_boundary_with_pca(X, y):
    """
    Applies PCA to reduce features to 2 dimensions and plots the data points and decision boundary of a decision tree classifier,
    with accuracy displayed on the plot.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
    """
    # Apply PCA to reduce the features to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Initialize and train the decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_pca, y)

    # Calculate the accuracy
    accuracy = accuracy_score(y, model.predict(X_pca))

    plt.figure(figsize=(10, 6))

    # Plot the data points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('32561 people in the United States\nReducing 14 features to 2 features by using PCA')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Under 50K', 'Over 50K'])

    # Plot the decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)

    # Display the accuracy on the plot
    plt.text(x_max - 0.5, y_min + 0.5, f'Accuracy: {accuracy:.2f}',
             horizontalalignment='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    plt.show()


def plot_accuracy(scores, title):
    """
    Plots the cross-validation or random split scores.

    Args:
        scores (numpy.ndarray): Accuracy scores.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))

    # Plot the scores with more digits after the decimal point
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')

    # Add text annotations with more digits
    for i, score in enumerate(scores):
        plt.text(i + 1, score, f'{score:.5f}', ha='center', va='bottom')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0.8, 0.9)  # Adjust the limits to zoom in on relevant values
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Process the data
    X, y = process_data()

    # Plot decision boundary with PCA
    plot_decision_boundary_with_pca(X, y)

    # Perform Cross Validation
    avg_accuracy_cv = cross_validation(X, y)
    print(f"Average accuracy from Cross Validation (K-Fold): {avg_accuracy_cv:.5f}")

    # Perform Random Split
    avg_accuracy_random_split = random_split(X, y)
    print(f"Average accuracy from Random 50% Split: {avg_accuracy_random_split:.5f}")

    # Perform AdaBoost with Decision Tree
    ada_accuracy = adaboost(X, y, n_estimators=200)
    print(f"Accuracy from AdaBoost with Decision Tree: {ada_accuracy:.5f}")
