import os
import sys
import pandas as pd
from utils.features import *  # Import necessary functions from features module
from utils.utils import crop_img  # Import crop_img function from utils module
from tqdm import tqdm
import cv2

def main(data_directory, processed_direct, width, height):
    """
    Main function to process and resize images from training and testing datasets.

    Parameters:
        data_directory (str): Path to the directory containing the train and test dataset CSV files.
        processed_direct (str): Path to the directory where processed images will be saved.
        width (int): The desired width of the resized images.
        height (int): The desired height of the resized images.
    """
    
    # Read the train and test datasets
    df_train = pd.read_csv(os.path.join(data_directory, "train_dataset.csv"), sep="|")
    df_test = pd.read_csv(os.path.join(data_directory, "test_dataset.csv"), sep="|")

    new_train = []
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(row['path']))
        
        if image is None:
            print("Error: Unable to read image at", row['path'])
            continue  # Skip this image if it cannot be read
        
        # Crop the image
        cropped_image = crop_img(image)

        # Resize the image
        resized_image = cv2.resize(cropped_image, (width, height))
        
        # Save the resized image to the processed directory
        image_name = os.path.basename(row['path'])  # Extract the image name
        save_path = os.path.join(processed_direct, "train", image_name)  # Construct the save path
        cv2.imwrite(save_path, resized_image)  # Save the resized image
        
        # Update the image path in the row and append to new_train list
        new_row = row.copy()
        new_row["path"] = save_path
        new_train.append(new_row)

    new_test = []
    # Iterate over the rows of the test dataset
    for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(row['path']))

        if image is None:
            print("Error: Unable to read image at", row['path'])
            continue  # Skip this image if it cannot be read
        
        # Crop the image
        cropped_image = crop_img(image)

        # Resize the image
        resized_image = cv2.resize(cropped_image, (width, height))
        
        # Save the resized image to the processed directory
        image_name = os.path.basename(row['path'])  # Extract the image name
        save_path = os.path.join(processed_direct, "test", image_name)  # Construct the save path
        cv2.imwrite(save_path, resized_image)  # Save the resized image
        
        # Update the image path in the row and append to new_test list
        new_row = row.copy()
        new_row["path"] = save_path
        new_test.append(new_row)

    # Convert lists to pandas DataFrames
    new_df_train = pd.DataFrame(new_train)
    new_df_test = pd.DataFrame(new_test)

    # Save the new DataFrames to CSV files
    new_df_train.to_csv(os.path.join(processed_direct, "train_dataset.csv"), sep="|", index=False)
    new_df_test.to_csv(os.path.join(processed_direct, "test_dataset.csv"), sep="|", index=False)

if __name__ == "__main__":
    # Change working directory to the project directory
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")  # Adapt this to the location of the PROJECT directory
    sys.path.append("CODEDESCR_ML")  # Add the path to the utils module to the Python path

    # Define paths for data directories and image dimensions
    data_directory = r"DATA\datasets"
    processed_direct = r"DATA\processed"

    # Execute main function
    main(data_directory, processed_direct, 300, 300)
