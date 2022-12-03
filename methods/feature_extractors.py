import tensorflow as tf


#### resnet50_untrained
def resnet50_pretrained(config):
    # load trained model
    feature_extractor = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=config["image_size"].append(3), 
    )

    # freeze weigths, use it only as inference
    for layer in feature_extractor.layers:
        layer.trainable = False

    return feature_extractor


#### resnet50_trained
def sift(config):
    pass


#### our_cnn_trained
def our_cnn_trained(config):
    # load trained model
    trained_model = tf.keras.models.load_model(config["our_trained_model_path"])

    # freeze weigths, use it only as inference
    for layer in trained_model.layers:
        layer.trainable = False

    #get the first part of the model to the flatten layer
    feature_extractor = trained_model.layers[:-2]


    return feature_extractor


