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
    """
    Main function to perform PCA and KMeans clustering on image features extracted using ResNet,
    then save the clustered images and visualize them using TensorBoard. The clusters are expected to contain
    only IRM with the same plan (axial, sagital or coronal). Each cluster of image is saved in its own folder.
    Later, the unclean folder are cleaned by hand.

    Parameters:
        processed_direct (str): Directory where the processed dataset with ResNet features is saved.
        save_cluster_dir (str): Directory where the clustered images will be saved.
    """

    # Read the data
    resnet_df_train = pd.read_pickle(os.path.join(processed_direct, "train_dataset_feature_resnet.pkl"))
    resnet_df_test = pd.read_pickle(os.path.join(processed_direct, "test_dataset_feature_resnet.pkl"))
    resnet_df = pd.concat([resnet_df_train, resnet_df_test])

    # Extract features for all images in the directory
    features = resnet_df.drop(columns=["path", "filename", "label"])

    # Apply PCA
    pca = PCA(n_components = 0.97)  # Retain 97% of variance
    pca_result = pca.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=15)  # Number of clusters
    kmeans.fit(pca_result)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Add cluster labels as a new column in the dataframe
    clustered_df = resnet_df[["path", "filename", "label"]]
    clustered_df['cluster'] = cluster_labels

    # Save the dataframe with cluster labels
    for index, row in clustered_df.iterrows():
        label = str(row['label'])
        cluster = str(row['cluster'])
        source_path = row['path']
        destination_dir = os.path.join(save_cluster_dir, label, cluster)
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy(source_path, destination_dir)

    # Create a summary writer
    log_dir = os.path.join(save_cluster_dir, "log_dir")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Sample some images from each cluster and create a grid
    for cluster_id in range(15):  # Assuming 15 clusters
        sample_images = clustered_df[clustered_df['cluster'] == cluster_id].sample(20)
        image_grid = []
        for index, row in sample_images.iterrows():
            image_path = row['path']
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((100, 100))
            image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1))
            image_grid.append(image_tensor)
        image_grid = make_grid(image_grid, nrow=8)

        # Add the image grid to TensorBoard with cluster label as title
        writer.add_image(f'Cluster_{cluster_id}', image_grid, dataformats='CHW')

    # Close the summary writer
    writer.close()

if __name__ == "__main__":
    # Change directory to the project directory
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") 
    sys.path.append("CODE\DESCR_ML")

    processed_direct = r"DATA\processed"
    save_cluster_dir = r"DATA\clustered_resnet"

    main(processed_direct, save_cluster_dir)