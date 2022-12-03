"""
This script trains the choosen classifier.

                                    trains this
                                         |
          ___________________       _____|______
data --> | feature extractor | --> | classifier | --> predictions
         |___________________|     |____________|

"""


import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow.keras import layers
import yaml
import joblib
import wandb
from wandb.keras import WandbCallback
from utils_classificator import get_dataset_partitions_tf
from datetime import datetime
import numpy as np

from methods.initalize_method import get_feature_extractor
from methods.initalize_method import get_classifier


# read in config file
with open("config_train_classifier.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# init wandb logger
timestamp = str(datetime.now()) # run identifier

if config["classifier"] == "mlp":
    wandb.init(
        project="BSC_THESIS",
        entity="zjarak",
        name=f"run_{timestamp}",
        reinit=True,
        dir="logs",
        save_code=True,
    )

    # save hyperparameters
    wandb.config.update(config)


# get data
data = tf.keras.preprocessing.image_dataset_from_directory(
    config["input_folder"],
    #label_mode="categorical",
    #validation_split=config["validation_split"],
    #subset="both",
    seed=1337,
    image_size=config["image_size"],
    batch_size=config["batch_size"],
    #shuffle=config["shuffle"]
)

# split data
train_ds, val_ds = get_dataset_partitions_tf(
    data,
    ds_size=len(data),
    train_split=config["train_split"],
    val_split=config["validation_split"],
)

num_of_classes = 8



# BUILD MODEL FORM FEATURE_EXTRACTOR + CLASSIFIER

######################################################################### 0. FEATURE EXTRACTOR <-- CNN
if (config["feature_extractor"] in  ["our_cnn_trained", "resnet50_pretrained"]):
    
    #initalize feature extractor
    feature_extractor = get_feature_extractor(config["feature_extractor"] , config)
    feature_vector_length = feature_extractor[-1].output.shape[-1]
    print(f"Feature vector length: {feature_vector_length}")
#----------------------------------------------------------------------- 1. CLASSIFIER <-- MLP
    if config["classifier"] == "mlp":
        classifier = get_classifier("mlp", config, feature_vector_length, num_of_classes)
        model_structure = feature_extractor + classifier
        full_model = tf.keras.Sequential(model_structure)

#----------------------------------------------------------------------- 1. CLASSIFIER <-- OTHERS (SKLEARN)
    else:
        feature_extractor = tf.keras.Sequential(feature_extractor)
        classifier = get_classifier(config["classifier"], config)
######################################################################### FEATURE EXTRACTOR <-- SIFT
elif config["feature_extractor"] == "sift":
    raise f"Not implemented yet."
else:
    raise f"Your {config['feature_extractor']} feature extractor is not implemented."






##################################### TRAIN WITH TENSORFLOW
if config["classifier"] == "mlp":
    # define callbacks and compiler
    callbacks = [
        WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=True,
            save_graph=True,
        )]

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # train classifier
    full_model.fit(
        train_ds,
        epochs=config["nr_of_epochs"],
        callbacks=callbacks,
        validation_data=val_ds
    )

##################################### TRAIN WITH SKLEARN
else:
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # generate dataset for sklearn
    for train_batch in train_ds:
        temp_batch = feature_extractor.predict(train_batch[0])
        X_train.append(np.array(temp_batch))
        y_train.append(np.array(train_batch[1]))
    
    for validation_batch in train_ds:
        temp_batch = feature_extractor.predict(validation_batch[0])
        X_test.append(np.array(temp_batch))
        y_test.append(np.array(validation_batch[1]))

    X_train = np.reshape(X_train, (-1,feature_vector_length))
    y_train = np.reshape(y_train, (-1,1)).squeeze()
    X_test = np.reshape(X_test, (-1,feature_vector_length))
    y_test = np.reshape(y_test, (-1,1)).squeeze()

    # train classifier
    classifier.fit(X_train, y_train)

    # test on validation set
    mean_accuracy = classifier.score(X_test,y_test)
    print(f"Mean accuracy of [{config['feature_extractor']}] -> [{config['classifier']}] model: {mean_accuracy} ")

    # save the trained classifier model
    joblib.dump(classifier, f"trained_models/classifiers/{config['classifier']}_trained.joblib")
