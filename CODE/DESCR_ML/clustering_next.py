import pandas as pd
import os
import sys
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


def main(processed_direct, save_cluster_dir):
    # Read the data
    resnet_df_train = pd.read_pickle(os.path.join(processed_direct, "train_dataset_feature_resnet.pkl"))
    resnet_df_test = pd.read_pickle(os.path.join(processed_direct, "test_dataset_feature_resnet.pkl"))
    resnet_df = pd.concat([resnet_df_train, resnet_df_test])

    # Extract features for all images in the directory
    features = resnet_df.drop(columns=["path", "filename", "label"])

    # Apply PCA
    pca = PCA(n_components = 0.97)  # Adjust as needed
    pca_result = pca.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=15)  # Adjust as needed
    kmeans.fit(pca_result)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Add cluster labels as a new column in the dataframe
    clustered_df = resnet_df[["path", "filename", "label"]]
    clustered_df['cluster'] = cluster_labels

    # Save the dataframe with cluster labels
     # Iterate through each row in the dataframe
    for index, row in clustered_df.iterrows():
        # Extract necessary information from the dataframe
        label = str(row['label'])
        cluster = str(row['cluster'])
        source_path = row['path']
        destination_dir = os.path.join(save_cluster_dir, label, cluster)

        # Create the directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Copy the image to the destination directory
        shutil.copy(source_path, destination_dir)

    # Create a summary writer
    log_dir = os.makedirs(os.path.join(save_cluster_dir, "log_dir"), exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Sample some images from each cluster and create a grid
    for cluster_id in range(15):  # Assuming 3 clusters
        # Get sample images from the cluster
        sample_images = clustered_df[clustered_df['cluster'] == cluster_id].sample(20)

        # Load, resize, and concatenate the images into a grid
        image_grid = []
        for index, row in sample_images.iterrows():
            image_path = row['path']
            image = Image.open(image_path)  # Load image
            if image.mode != 'RGB':  # Convert grayscale to RGB
                image = image.convert('RGB')
            image = image.resize((100, 100))  # Resize to desired size
            image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1))  # Convert to tensor (CHW format)
            image_grid.append(image_tensor)
        image_grid = make_grid(image_grid, nrow=8)  # Create a grid with 3 columns

        # Add the image grid to TensorBoard with cluster label as title
        writer.add_image(f'Cluster_{cluster_id}', image_grid, dataformats='CHW')

    # Close the summary writer
    writer.close()

if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODE\DESCR_ML")

    processed_direct = r"DATA\processed"
    save_cluster_dir = r"DATA\clustered_resnet"

    main(processed_direct, save_cluster_dir)
