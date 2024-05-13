import os
import sys
import pandas as pd
from utils.features import *
from tqdm import tqdm
import shutil


l = []

def add_section_columns(df:pd.DataFrame, axial_images, coronal_images, sagittal_images):
    section = []

    for index, row in tqdm(df.iterrows()):
        filename =row["filename"]
        new_row = row
        if filename in axial_images:
            new_row["axial"] = 1
            new_row["corona"] = 0
            new_row["sagittal"] = 0
        elif filename in coronal_images:
            new_row["axial"] = 0
            new_row["corona"] = 1
            new_row["sagittal"] = 0
        elif filename in sagittal_images:
            new_row["axial"] = 0
            new_row["corona"] = 0
            new_row["sagittal"] = 1

        section.append(new_row)
    return section


def main(processed_direct, dataset_directory, axial_dir, corona_dir, sagital_dir):
    # Read the train and test datasets
    df_train = pd.read_csv(os.path.join(dataset_directory, "train_dataset.csv"), sep="|")
    df_test = pd.read_csv(os.path.join(dataset_directory, "test_dataset.csv"), sep="|")

    axial_images = os.listdir(axial_dir)
    corona_imges = os.listdir(corona_dir)
    sagittal_imges = os.listdir(sagital_dir)

    print(len(axial_images) + len(corona_imges) + len(sagittal_imges))
    print()
    print(len(df_test) + len(df_train))


    train_section = add_section_columns(df_train, axial_images, corona_imges, sagittal_imges)
    test_section = add_section_columns(df_test, axial_images, corona_imges, sagittal_imges)

    
    # sys.exit()

    df_train = pd.DataFrame(train_section)
    df_test = pd.DataFrame(test_section)


    hog = HOG()
    lbp = LocalBinaryPatterns(numPoints=24, radius=8)
    lab_histogram = LabHistogram()
    color_channel_statistic = ColorChannelStatistic()

    new_train = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the train dataset
    for index, row in tqdm(df_train.iterrows()):
        # Read the image using OpenCV
        image_path = row['path']

        # Extract features using different methods
        color_features = color_channel_statistic.describe(image_path)
        lab_features = lab_histogram.describe(image_path)  # Only taking the histogram part
        lbp_features = lbp.describe(image_path)  # Only taking the histogram part
        hog_features = hog.describe(image_path)  # Only taking the HOG features

        new_train["color"].append(color_features)
        new_train["lab"].append(lab_features)
        new_train["lbp"].append(lbp_features)
        new_train["hog"].append(hog_features)
        

    new_test = {"color": [], "lab": [], "lbp": [], "hog": [], "resnet": []}
    # Iterate over the rows of the test dataset
    for index, row in tqdm(df_test.iterrows()):
        # Read the image using OpenCV
        image_path = row['path']

        # Extract features using different methods
        color_features = color_channel_statistic.describe(image_path)
        lab_features = lab_histogram.describe(image_path)  # Only taking the histogram part
        lbp_features = lbp.describe(image_path)  # Only taking the histogram part
        hog_features = hog.describe(image_path)  # Only taking the HOG features

        new_test["color"].append(color_features)
        new_test["lab"].append(lab_features)
        new_test["lbp"].append(lbp_features)
        new_test["hog"].append(hog_features)

    # Convert lists of features to dataframes
    color_df_train = pd.DataFrame(new_train["color"], columns=[f'color_feature_{i}' for i in range(len(new_train["color"][0]))])
    lab_df_train = pd.DataFrame(new_train["lab"], columns=[f'lab_feature_{i}' for i in range(len(new_train["lab"][0]))])
    lbp_df_train = pd.DataFrame(new_train["lbp"], columns=[f'lbp_feature_{i}' for i in range(len(new_train["lbp"][0]))])
    hog_df_train = pd.DataFrame(new_train["hog"], columns=[f'hog_feature_{i}' for i in range(len(new_train["hog"][0]))])

    color_df_train = pd.concat([color_df_train, df_train], axis=1)
    lab_df_train = pd.concat([lab_df_train, df_train], axis=1)
    lbp_df_train = pd.concat([lbp_df_train, df_train], axis=1)
    hog_df_train = pd.concat([hog_df_train, df_train], axis=1)


    color_df_test = pd.DataFrame(new_test["color"], columns=[f'color_feature_{i}' for i in range(len(new_test["color"][0]))])
    lab_df_test = pd.DataFrame(new_test["lab"], columns=[f'lab_feature_{i}' for i in range(len(new_test["lab"][0]))])
    lbp_df_test = pd.DataFrame(new_test["lbp"], columns=[f'lbp_feature_{i}' for i in range(len(new_test["lbp"][0]))])
    hog_df_test = pd.DataFrame(new_test["hog"], columns=[f'hog_feature_{i}' for i in range(len(new_test["hog"][0]))])

    color_df_test = pd.concat([color_df_test, df_test], axis=1)
    lab_df_test = pd.concat([lab_df_test, df_test], axis=1)
    lbp_df_test = pd.concat([lbp_df_test, df_test], axis=1)
    hog_df_test = pd.concat([hog_df_test, df_test], axis=1)

    color_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_color.pkl"))
    lab_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_lab.pkl"))
    lbp_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_lbp.pkl"))
    hog_df_train.to_pickle(os.path.join(processed_direct, "train_dataset_feature_hog.pkl"))

    color_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_color.pkl"))
    lab_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_lab.pkl"))
    lbp_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_lbp.pkl"))
    hog_df_test.to_pickle(os.path.join(processed_direct, "test_dataset_feature_hog.pkl"))


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")
    sys.path.append("CODE")

    processed_direct = r"DATA\processed"
    dataset_directory = r"DATA\datasets"
    axial_dir = r"DATA\clustered_section\clustered_resnet\axial"
    corona_dir = r"DATA\clustered_section\clustered_resnet\coronal"
    sagital_dir = r"DATA\clustered_section\clustered_resnet\sagittal"

    main(processed_direct, processed_direct, axial_dir, corona_dir, sagital_dir)
    print(l)



