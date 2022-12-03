"""
This script trains the choosen feature extractor.

            trains this
                |
          ______|____________       ____________
data --> | feature extractor | --> | classifier | --> predictions
         |___________________|     |____________|

"""


import tensorflow as tf
from tensorflow.keras import layers
import yaml
import wandb
from wandb.keras import WandbCallback
from utils_classificator import get_dataset_partitions_tf
from datetime import datetime


# read in config file
with open("config_train_feature_extractor.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# init wandb logger
timestamp = str(datetime.now()) # run identifier
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
train_ds, val_ds= get_dataset_partitions_tf(
    data,
    ds_size=len(data),
    train_split=config["train_split"],
    val_split=config["validation_split"],
)

# build model
num_of_classes = 8

if config["feature_extractor"] == "our_cnn":
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_of_classes)
    ])
else:
    raise f"Your {config['feature_extractor']} feature extractor is not implemented."

# define callbacks and compiler
callbacks = [
    WandbCallback(
        monitor="val_loss",
        mode="min",
        save_model=True,
        save_graph=True,
    )]

model.compile(
    optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=config["nr_of_epochs"],
    callbacks=callbacks,
    validation_data=val_ds
)