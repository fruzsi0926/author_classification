"""
This script is for initialize different model parts
such as, different feature extractor (sift, cnn, ...) and
classifier (mlp, naive bayes, ...).
"""

# feature extractors
from methods.feature_extractors import resnet50_pretrained
from methods.feature_extractors import our_cnn_trained

# classifiers
from methods.classifiers import mlp
from methods.classifiers import linear_svm
from methods.classifiers import k_nn
from methods.classifiers import rbf_svm
from methods.classifiers import decision_tree



def get_feature_extractor(extractor_type: str, config):
    if extractor_type == "resnet50_pretrained":
        return resnet50_pretrained(config)
    elif extractor_type == "sift":
        return None
    elif extractor_type == "our_cnn_trained":
        return our_cnn_trained(config)
    else:
        raise f"{extractor_type} is not implemented feature_extractor type."

def get_classifier(classifier_type: str, config, feature_vector_length: int = None, num_of_classes: int = None):
    if classifier_type == "mlp":
        return mlp(feature_vector_length, num_of_classes)
    elif classifier_type == "linear_svm":
        return linear_svm(config)
    elif classifier_type == "rbf_svm":
        return rbf_svm(config)
    elif classifier_type == "decision_tree":
        return decision_tree(config)
    elif classifier_type == "random_forest":
        return resnet50_trained(config)
    elif classifier_type == "naive_bayes":
        return None
    elif classifier_type == "qda":
        return our_cnn_trained(config)
    elif classifier_type == "k_nn":
        return k_nn(config)
    else:
        raise f"{classifier_type} is not implemented classifier type."