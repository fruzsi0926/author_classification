"""
This script test the different models.

                                                            gives this result
                                                                     |                                                           
          ________________________________                           |
data --> | feature extractor | classifier | --> count votes --> predicted author
         |___________________|____________|                     '''''''''''''''
"""

from utils_dataset import DatasetGenerator
import tempfile
import tensorflow as tf
import yaml
import joblib
import numpy as np

from methods.initalize_method import get_feature_extractor
from methods.initalize_method import get_classifier


# read in test config file
with open("config_test.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


correct_labels_of_test_data = {
    "test_0": config["test_0"],
    "test_1": config["test_1"],
}

model_prediction_mapping = [
    "Bajza_Jozsef",
    "Guzmics_Izidor",
    "Kazinczy_Ferenc",
    "Kazinczy_Gabor"
]



# class for data preprocessing
test_data_generator = DatasetGenerator(img_folder="dataset/raw_data_test")

# everything is done in a temp folder
temp_folder_name = f"temp_test_folder_{config['image_style']}"
with tempfile.TemporaryDirectory() as tmpdirname:
    print('Created temporary directory: ', tmpdirname)

    # process images for CNN
    test_data_generator.generate_test_dataset(
        img_style=config['image_style'],
        desired_size=config["image_size"],
        temp_folder_name=temp_folder_name,
        )

    # load processed data for CNN
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    temp_folder_name,
    seed=1337,
    image_size=config["image_size"],
    batch_size=None, #no batching
    shuffle=False
    )

    # BUILD MODEL FORM FEATURE_EXTRACTOR + CLASSIFIER

    ######################################################################### 0. FEATURE EXTRACTOR <-- CNN
    if (config["feature_extractor"] in  ["our_cnn_trained", "resnet50_pretrained"]):
        
        #initalize feature extractor
        feature_extractor = get_feature_extractor(config["feature_extractor"] , config)
        feature_vector_length = feature_extractor[-1].output.shape[-1]
        print(f"Feature vector length: {feature_vector_length}")
    #----------------------------------------------------------------------- 1. CLASSIFIER <-- MLP
        if config["classifier"] == "mlp":
            classifier = tf.keras.models.load_model(config["mlp_trained_path"]).layers[-2:]
            model_structure = feature_extractor + classifier
            full_model = tf.keras.Sequential(model_structure)

    #----------------------------------------------------------------------- 1. CLASSIFIER <-- OTHERS (SKLEARN)
        else:
            feature_extractor = tf.keras.Sequential(feature_extractor)
            classifier = joblib.load(config[f"{config['classifier']}_trained_path"])
    ######################################################################### FEATURE EXTRACTOR <-- SIFT
    elif config["feature_extractor"] == "sift":
        raise f"Not implemented yet."
    else:
        raise f"Your {config['feature_extractor']} feature extractor is not implemented."

    

    test_labels = test_ds.class_names
    data_by_images = {}

    for label in test_labels:
        data_by_images[label] = []

    for input, label in test_ds:
        label = f"test_{np.asarray(label)}"
        data_by_images[label].append(input)


    final_predictions = {}
    # predictions for one picture:
    for each_image_name in data_by_images.keys():

        # if tensorflow model is used
        if config["classifier"] == "mlp":
            predictions = full_model.predict(np.array(data_by_images[each_image_name]))
            predictions = np.argmax(predictions, axis=1)
        
        # if sklearn model is used
        else:
            feature_vectors = feature_extractor.predict(np.array(data_by_images[each_image_name]))
            predictions = classifier.predict(feature_vectors)

        # calculate votes for each authors
        nr_of_occurences= np.bincount(predictions)
        print(f"### The prediction for the -> {each_image_name} <- image ###")
        for idx, votes in enumerate(nr_of_occurences):
            print(f"For {model_prediction_mapping[idx]} number of vote: {votes}")
        winner = model_prediction_mapping[np.argmax(nr_of_occurences)]
        print(f"The predicted author is {winner}!")

        final_predictions[each_image_name] = winner


    print(f"\n|-        -|-    ground_truth    -|-    prediciton     -| ")
    for test_image_name in final_predictions.keys():
        print(f"| {test_image_name} ->  {correct_labels_of_test_data[test_image_name]} -> {final_predictions[test_image_name]}")


print("Finished testing.")