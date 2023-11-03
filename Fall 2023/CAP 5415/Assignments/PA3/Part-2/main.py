from sklearn import datasets
from classifier import KNNClassifier
import numpy as np

def test_knn(k, X_train, X_test, y_train, y_test):
    classifier = KNNClassifier(k=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print("k = ", k, " accuracy = ", accuracy)
    return accuracy

def main():
    digits = datasets.load_digits()
    X = digits.data  # Features (images)
    y = digits.target  # Target labels (digit class)

    # Split the dataset into training and testing sets
    selected_data = []
    selected_labels = []

    # Define the number of images to select per class (50 in this case)
    images_per_class_testing = 50

    # Iterate over each digit class (0 to 9)
    for digit in range(10):
        # Find the indices of all instances of the current digit
        indices = np.where(y == digit)[0]

        # Randomly select 50 indices from the current digit class for testing
        selected_indices = np.random.choice(indices, size=images_per_class_testing, replace=False)

        # Append the selected data and labels to the lists
        selected_data.extend(X[selected_indices])
        selected_labels.extend(y[selected_indices])

    # Convert the selected data and labels to NumPy arrays for testing
    X_test = np.array(selected_data)
    y_test = np.array(selected_labels)

    # Remove the selected testing data from the original data to get the remaining training data
    remaining_indices = np.setdiff1d(np.arange(len(X)), selected_indices)
    # if remaining_indices.ndim == 0:
    #     remaining_indices = np.array([remaining_indices])
    X_train = X[remaining_indices]
    y_train = y[remaining_indices]

    accuracy = []
    neighbors = [3, 5, 7]
    for k in neighbors:
        accuracy_out = test_knn(k, X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()