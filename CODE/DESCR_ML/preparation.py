import os
import pandas as pd

def main(train_data_directory, test_data_directory, save_directory):
    """
    Main function to generate CSV files for training and testing datasets.
    
    Parameters:
        train_data_directory (str): Path to the directory containing training images.
        test_data_directory (str): Path to the directory containing testing images.
        save_directory (str): Path to the directory where the generated CSV files will be saved.
    
    Note:
        The glioma class images have been removed based on the dataset owner's recommendation.
    """
    
    # Labels for the dataset (excluding glioma as recommended)
    labels = ["notumor", "meningioma", "pituitary"]

    # Initialize lists to store image information
    test = []
    train = []

    # Process training images
    for label in labels:
        directory = os.path.join(train_data_directory, label)
        image_list = os.listdir(directory)
        for filename in image_list[:]:
            if filename.endswith('.jpg'):  # Process only .jpg files
                row = {"filename": filename, "path": os.path.join(directory, filename), "label": label}
                train.append(row)

    # Process testing images
    for label in labels:
        directory = os.path.join(test_data_directory, label)
        image_list = os.listdir(directory)
        for filename in image_list[:]:
            if filename.endswith('.jpg'):  # Process only .jpg files
                row = {"filename": filename, "path": os.path.join(directory, filename), "label": label}
                test.append(row)

    # Convert lists to pandas DataFrames
    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)

    # Save DataFrames to CSV files
    df_train.to_csv(os.path.join(save_directory, "train_dataset.csv"), sep="|", index=False)
    df_test.to_csv(os.path.join(save_directory, "test_dataset.csv"), sep="|", index=False)

if __name__ == "__main__":
    # Change working directory
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")  # Adapt this to the location of the PROJECT directory

    # Define paths for training and testing data directories and save directory
    train_data_directory = r"DATA\Training"
    test_data_directory = r"DATA\Testing"
    save_directory = r"DATA\datasets"

    # Execute main function
    main(train_data_directory, test_data_directory, save_directory)
