import os
import sys
import pandas as pd
from utils.features import *
from utils.utils import  crop_img
from tqdm import tqdm


def main(data_directory, processed_direct, width, height):
    # Read the train and test datasets
    df_train = pd.read_csv(os.path.join(data_directory, "train_dataset.csv"), sep="|")
    df_test = pd.read_csv(os.path.join(data_directory, "test_dataset.csv"), sep="|")

    new_train = []
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_train.iterrows()):
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(row['path']))
        
        if image is None:
            print("Error: Unable to read image at", row['path'])
            return None
        # crop image
        croped_image = crop_img(image)

        # Resize the image (adjust the dimensions as needed)
        resized_image = cv2.resize(croped_image, (width, height))  # Specify the new dimensions
        
        # Save the resized image to the processed directory
        image_name = os.path.basename(row['path'])  # Extract the image name
        save_path = os.path.join(processed_direct, "train", image_name)  # Construct the save path
        cv2.imwrite(save_path, resized_image)  # Save the resized image
        new_row = row
        new_row["path"] = save_path
        new_train.append(new_row)

    new_test = []
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_test.iterrows()):
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(row['path']))

        if image is None:
            print("Error: Unable to read image at", row['path'])
            return None
        # crop image
        croped_image = crop_img(image)

        # Resize the image (adjust the dimensions as needed)
        resized_image = cv2.resize(croped_image, (width, height))  # Specify the new dimensions
        
        # Save the resized image to the processed directory
        image_name = os.path.basename(row['path'])  # Extract the image name
        save_path = os.path.join(processed_direct, "test", image_name)  # Construct the save path
        cv2.imwrite(save_path, resized_image)  # Save the resized image
        new_row = row
        new_row["path"] = save_path
        new_test.append(new_row)

    new_df_train = pd.DataFrame(new_train)
    new_df_test = pd.DataFrame(new_test)

    new_df_train.to_csv(os.path.join(processed_direct, "train_dataset.csv"), sep = "|", index = False)
    new_df_test.to_csv(os.path.join(processed_direct, "test_dataset.csv"), sep = "|", index = False)


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODEDESCR_ML")

    data_directory = r"DATA\datasets"
    processed_direct = r"DATA\processed"

    main(data_directory, processed_direct, 300, 300)