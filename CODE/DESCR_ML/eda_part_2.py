import os
import sys
import pandas as pd
from utils.features import *
from utils.utils import  crop_img
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import json
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import random

def train_and_evaluate_classifier(df_train, df_test, classifier, save_dir, columns_to_drop = ["filename", "path"]):
    X_train, y_train, X_test, y_test, label_encoder = prepare_data(df_train, df_test, columns_to_drop)
    writer = SummaryWriter(os.path.join(save_dir, "runs"))
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep components that explain 95% of variance

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)

    # Plot the explained variance ratio
    plot_variance_explained(pca, save_dir)
    plot_feature_participation_before_pca(X_train, pca, save_dir)

    X_test_pca = pca.transform(X_test_scaled)

    # Train classifier
    classifier.fit(X_train_pca, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test_pca)
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Randomly select 20 images for each predicted label
    if "hog_img_path" in df_test.columns or "lab_hist_path" in df_test.columns:
        for label in set(y_pred_original):
            for plan in set(df_test['plan']):
                indices = [i for i, (y, p) in enumerate(zip(y_pred_original, df_test['plan'])) if y == label and p == plan]
                selected_indices = random.sample(indices, min(20, len(indices)))
                if "hog_img_path" in df_test.columns:
                    selected_image_hog_paths = df_test.loc[selected_indices, 'hog_img_path'].tolist()
                    tag_lab = f'HOG_Histtogram_Label_{label}_Plan_{plan}'
                    add_images_writer(selected_image_hog_paths, writer, tag_lab)
                if "lab_hist_path" in df_test.columns:
                    selected_image_lab_paths = df_test.loc[selected_indices, 'lab_hist_path'].tolist()
                    tag_hog = f'LAB_Histtogram_Label_{label}_Plan_{plan}'
                    add_images_writer(selected_image_lab_paths, writer, tag_hog)
    

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)

    # Parse the classification report string
    report_lines = report.split('\n')
    class_metrics = {}
    for line in report_lines[2:-5]:  # Skip header and footer lines
        tokens = line.split()
        class_label = label_encoder.inverse_transform([int(tokens[0])])[0]
        metrics = {
            "Precision": float(tokens[1]),
            "Recall": float(tokens[2]),
            "F1 Score": float(tokens[3]),
            "Support": int(tokens[4])
        }
        class_metrics[class_label] = metrics

    global_metrics = {}
    for i, line in enumerate(report_lines[-4:-1]):  # Extract global accuracy, macro avg, and weighted avg
        tokens = line.split()
        if i == 0 and tokens:
            metric_name = tokens[0]
            metric_value = {
                    "f1-score": float(tokens[1]),
                    "support": int(tokens[2])}
            global_metrics[metric_name] = metric_value
        elif tokens:
            metric_name = tokens[0] + " " + tokens[1]
            metric_value = {
                    "precision": float(tokens[2]),
                    "recall": float(tokens[3]),
                    "f1-score": float(tokens[4]),
                    "support": int(tokens[5])}
            global_metrics[metric_name] = metric_value

    class_report = {"Class metrics": class_metrics, "Global metrics": global_metrics}
    # Store the metrics and report in a dictionary
    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Classification Report": class_report
    }

    # Save the dictionary to a JSON file
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels = np.unique(y_test)).plot()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)
    plt.tight_layout()

    # Save the confusion matrix plot as an image
    cm_image_path = os.path.join(save_dir, "confusion_matrix_plot.png")
    plt.savefig(cm_image_path)
    plt.close()

    # Load the image using PIL
    cm_image_pil = Image.open(cm_image_path)
    # Convert the PIL Image to a NumPy array
    cm_image_array = np.array(cm_image_pil)
    # Convert the data type of the image array to uint8
    cm_image_array = cm_image_array.astype(np.uint8)
    writer.add_image("Confusion Matrix Plot", cm_image_array, dataformats='HWC')

    # Add other metrics to TensorBoard
    writer.add_scalar("Accuracy", accuracy)
    writer.add_scalar("Precision", precision)
    writer.add_scalar("Recall", recall)
    writer.add_scalar("F1 Score", f1)


def prepare_data(df_train:pd.DataFrame, df_test:pd.DataFrame, columns_to_drop:list[str]):
    df_train = df_train.drop(columns = columns_to_drop)
    df_test = df_test.drop(columns = columns_to_drop)
    
    if "hog_img_path" in df_test.columns or "lab_hist_path" in df_test.columns:
        if "hog_img_path" in df_test.columns:
            df_train = df_train.drop(columns = ["hog_img_path"])
            df_test = df_test.drop(columns = ["hog_img_path"]) 
        if "lab_hist_path" in df_test.columns:
            df_train = df_train.drop(columns = ["lab_hist_path"])
            df_test = df_test.drop(columns = ["lab_hist_path"]) 

    # Separate features (X) and labels (y)
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Apply label encoding to convert text labels to numeric representations
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train, y_train_encoded, X_test, y_test_encoded, label_encoder

def plot_variance_explained(pca, save_dir):
    # Calculate the cumulative sum of explained variance
    explained_variance_ratio_cumsum = pca.explained_variance_ratio_.cumsum()

    # Plot the cumulative sum of explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(explained_variance_ratio_cumsum, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio by PCA Components')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'cumulate_explained_variance_ratio.png'))  # Save the plot as an image

def plot_feature_participation_before_pca(X_train, pca, save_dir):
    # Get the absolute values of the coefficients of the first two principal components
    PC1 = pca.components_[0]
    PC2 = pca.components_[1]

    # Get the feature names
    features = X_train.columns

    # Plot the participation of each feature in the first two principal components
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.barh(features, PC1)
    plt.title("PC1")
    plt.subplot(122)
    plt.barh(features, PC2)
    plt.title("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'explained_variance_ratio.png'))  # Save the plot as an image


def add_images_writer(image_paths, writer, tag):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(tag)
    axes = axes.flatten()
    for i, (path, ax) in enumerate(zip(image_paths, axes)):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
    writer.add_figure(tag, fig)

def main(data_directory):
    descriptors_set = ["lbp", 'hog', 'lab', "color"]

    # Initialize an empty list to store the combinations
    descriptor_combinations = []

    # Generate combinations of all lengths from 2 to the length of descriptors
    for r in range(1, len(descriptors_set) + 1):
        # Generate combinations of length r
        combinations_r = combinations(descriptors_set, r)
        
        # Extend the list of all combinations
        descriptor_combinations.extend(combinations_r)

    # Print all combinations
    for descriptors in descriptor_combinations:
        print(descriptors)

    for descriptors in descriptor_combinations:
        # Initialize an empty DataFrame to store the merged result
        merged_df_train = pd.DataFrame()
        merged_df_test = pd.DataFrame()

        # Iterate over each descriptor
        for descriptor in descriptors:
            # Read the train and test datasets for the current descriptor
            df_train = pd.read_pickle(os.path.join(data_directory, f"train_dataset_feature_{descriptor}.pkl"))
            df_test = pd.read_pickle(os.path.join(data_directory, f"test_dataset_feature_{descriptor}.pkl"))
            
            # Merge the DataFrames using the join method
            merged_df_train = pd.concat([merged_df_train, df_train.drop(columns = ["filename", "path", "label", "sagittal", "corona", "axial", "plan"])], axis=1)
            merged_df_test = pd.concat([merged_df_test, df_test.drop(columns = ["filename", "path", "label", "sagittal", "corona", "axial", "plan"])], axis=1)

            merged_df_train[["filename", "path", "label", "sagittal", "corona", "axial", "plan"]] = df_train[["filename", "path", "label", "sagittal", "corona", "axial", "plan"]]
            merged_df_test[["filename", "path", "label", "sagittal", "corona", "axial", "plan"]] = df_test[["filename", "path", "label", "sagittal", "corona", "axial", "plan"]]

        # Initialize and train a classifier
        classifier = RandomForestClassifier()

        # Define the directory to save logs
        # Create a summary writer
        descriptor_dir = '_'.join(descriptors)
        save_dir = os.path.join("LOGS", "tensorboard_no_plan", descriptor_dir)
        os.makedirs(save_dir, exist_ok=True)

        columns_to_drop_no_plan = ["filename", "path", "sagittal", "corona", "axial", "plan"]
        columns_to_drop_with_plan = ["filename", "path", "plan"]
        train_and_evaluate_classifier(merged_df_train, merged_df_test, classifier, save_dir, columns_to_drop = columns_to_drop_no_plan)

if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODE\DESCR_ML")

    processed_direct = r"DATA\processed"

    main(processed_direct)




