import tensorflow as tf
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

### MLP classifier
def mlp(feature_vector_length: int, num_of_classes: int):
    classifier = [
    tf.keras.layers.Dense(feature_vector_length, activation='relu'),
    tf.keras.layers.Dense(num_of_classes)
    ]

    return classifier


### linear SVM
def linear_svm(config: dict):
    classifier = SVC(
        kernel="linear",
        C=0.025
        )

    return classifier

### rbf SVM
def rbf_svm(config: dict):
    classifier = SVC(
        kernel="rbf",
        C=0.025
        )

    return classifier

### rbf SVM
def decision_tree(config: dict):
    classifier = tree.DecisionTreeClassifier()

    return classifier

### k_nn
def k_nn(config: dict):
    classifier = KNeighborsClassifier(n_neighbors=config["k"])
    
    return classifier