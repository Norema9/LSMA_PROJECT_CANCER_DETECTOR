import os
import sys
import pandas as pd
from utils.features import *
from tqdm import tqdm


def main(processed_direct, dataset_directory):
    # Read the train and test datasets
    df_train = pd.read_csv(os.path.join(dataset_directory, "train_dataset.csv"), sep="|")
    df_test = pd.read_csv(os.path.join(dataset_directory, "test_dataset.csv"), sep="|")

    resNetFeatureExtractor = ResNetFeature()

    new_train = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_train.iterrows()):
        # Read the image using OpenCV
        image_path = row['path']

        # Extract features using different methods
        resNetFeature = resNetFeatureExtractor.describe(image_path)
        new_train["resnet"].append(resNetFeature)

    new_test = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the test dataset
    for index, row in tqdm(df_test.iterrows()):
        # Read the image using OpenCV
        image_path = row['path']

        # Extract features using different methods
        resNetFeature = resNetFeatureExtractor.describe(image_path)
        new_test["resnet"].append(resNetFeature)

    # Convert lists of features to dataframes
    resnet_df_train = pd.DataFrame(new_train["resnet"], columns=[f'resnet_feature_{i}' for i in range(len(new_train["resnet"][0]))])
    resnet_df_train = pd.concat([resnet_df_train, df_train], axis=1)
    resnet_df_test = pd.DataFrame(new_test["resnet"], columns=[f'resnet_feature_{i}' for i in range(len(new_test["resnet"][0]))])
    resnet_df_test = pd.concat([resnet_df_test, df_test], axis=1)

    resnet_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_resnet.pkl"))
    resnet_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_resnet.pkl"))


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")
    sys.path.append("CODE")

    processed_direct = r"DATA\processed"
    dataset_directory = r"DATA\datasets"

    main(processed_direct, processed_direct)



