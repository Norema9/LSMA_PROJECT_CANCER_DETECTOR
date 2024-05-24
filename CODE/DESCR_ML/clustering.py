import os
import sys
import pandas as pd
from utils.features import ResNetFeature  # Import the ResNet feature extraction class from utils.features
from tqdm import tqdm  # Import tqdm for displaying a progress bar

def main(processed_direct, dataset_directory):
    """
    Main function to extract features from images in the training and testing datasets using ResNet and save the results.

    Parameters:
        processed_direct (str): Directory where the processed dataset with features will be saved.
        dataset_directory (str): Directory containing the CSV files for the train and test datasets.
    """
    
    # Read the train and test datasets
    df_train = pd.read_csv(os.path.join(dataset_directory, "train_dataset.csv"), sep="|")
    df_test = pd.read_csv(os.path.join(dataset_directory, "test_dataset.csv"), sep="|")

    # Initialize the ResNet feature extractor
    resNetFeatureExtractor = ResNetFeature()

    new_train = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
        # Get the image path
        image_path = row['path']

        # Extract features using ResNet
        resNetFeature = resNetFeatureExtractor.describe(image_path)
        new_train["resnet"].append(resNetFeature)

    new_test = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the test dataset
    for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # Get the image path
        image_path = row['path']

        # Extract features using ResNet
        resNetFeature = resNetFeatureExtractor.describe(image_path)
        new_test["resnet"].append(resNetFeature)

    # Convert lists of features to DataFrames and concatenate with the original DataFrames
    resnet_df_train = pd.DataFrame(new_train["resnet"], columns=[f'resnet_feature_{i}' for i in range(len(new_train["resnet"][0]))])
    resnet_df_train = pd.concat([resnet_df_train, df_train], axis=1)
    resnet_df_test = pd.DataFrame(new_test["resnet"], columns=[f'resnet_feature_{i}' for i in range(len(new_test["resnet"][0]))])
    resnet_df_test = pd.concat([resnet_df_test, df_test], axis=1)

    # Save the DataFrames with features to pickle files
    resnet_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_resnet.pkl"))
    resnet_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_resnet.pkl"))

if __name__ == "__main__":
    # Change working directory to the project directory
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")  # Adapt this to the location of the PROJECT directory
    sys.path.append(r"CODE\DESCR_ML")  # Add the path to the utils module to the Python path

    # Define paths for directories
    processed_direct = r"DATA\processed"
    dataset_directory = r"DATA\datasets"

    # Execute the main function
    main(processed_direct, dataset_directory)