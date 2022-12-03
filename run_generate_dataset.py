"""
This script generates the train dataset.
"""

from utils_dataset import DatasetGenerator
#GENERATE DATASET
# please delete regarding folders before run.

preprocess = DatasetGenerator(img_folder="dataset/raw_data_train")
preprocess.generate_train_dataset(img_style="processed")