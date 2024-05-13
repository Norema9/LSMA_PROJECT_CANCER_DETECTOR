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

def train_and_evaluate_classifier(df_train, df_test, classifier, save_dir, writer:SummaryWriter, columns_to_drop = ["filename", "path"]):
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test, columns_to_drop)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep components that explain 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train classifier
    classifier.fit(X_train_pca, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test_pca)

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

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test)).plot()
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


    # # Calculate precision and recall for each class
    # precision = dict()
    # recall = dict()
    # for i in range(len(classifier.classes_)):
    #     y_test_binary = label_binarize(y_test, classes=classifier.classes_)
    #     y_pred_binary = label_binarize(y_pred, classes=classifier.classes_)
    #     precision[i], recall[i], _ = precision_recall_curve(y_test_binary[:, i], y_pred_binary[:, i])

    # # Add precision and recall curves to the SummaryWriter for TensorBoard visualization
    # for i in range(len(classifier.classes_)):
    #     writer.add_scalar(f'Precision/Class_{i}', precision[i], global_step)
    #     writer.add_scalar(f'Recall/Class_{i}', recall[i], global_step)

    # # Compute precision-recall curve
    # precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # # Plot precision-recall curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall, precision, marker='.')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.grid(True)
    # plt.tight_layout()

    # # Save the precision-recall curve plot as an image
    # prc_image_path = os.path.join(save_dir, "precision_recall_curve_plot.png")
    # plt.savefig(prc_image_path)
    # plt.close()

    # # Add the precision-recall curve image to the SummaryWriter for TensorBoard visualization
    # # Load the image using PIL
    # prc_image_pil = Image.open(prc_image_path)
    # # Convert the PIL Image to a NumPy array
    # prc_image_array = np.array(prc_image_pil)# Load the image using PIL
    # # Convert the data type of the image array to uint8
    # prc_image_array = prc_image_array.astype(np.uint8)
    # writer.add_image("Precision-Recall Curve Plot", prc_image_array, dataformats='HWC')

    # Add other metrics to TensorBoard
    writer.add_scalar("Accuracy", accuracy)
    writer.add_scalar("Precision", precision)
    writer.add_scalar("Recall", recall)
    writer.add_scalar("F1 Score", f1)

def prepare_data(df_train:pd.DataFrame, df_test:pd.DataFrame, columns_to_drop:list[str]):
    df_train = df_train.drop(columns = columns_to_drop)
    df_test = df_test.drop(columns = columns_to_drop)
    # Separate features (X) and labels (y)
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Apply label encoding to convert text labels to numeric representations
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train, y_train_encoded, X_test, y_test_encoded


def main(data_directory):
    # Read the train and test datasets
    df_train = pd.read_pickle(os.path.join(data_directory, "train_dataset_feature_hog.pkl"))
    df_test = pd.read_pickle(os.path.join(data_directory, "test_dataset_feature_hog.pkl"))

    # print(df_test.columns)

    # Initialize and train a classifier
    classifier = RandomForestClassifier()

    # Define the directory to save logs
    save_dir = "LOGS"
    # Create a summary writer
    log_dir = os.makedirs(os.path.join(save_dir, "tensorboard_no_plan", "hog"), exist_ok=True)
    writer = SummaryWriter(log_dir)

    columns_to_drop_no_plan = ["filename", "path", "sagittal", "corona", "axial"]
    train_and_evaluate_classifier(df_train, df_test, classifier, save_dir, writer, columns_to_drop = columns_to_drop_no_plan)

if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")
    sys.path.append("CODE")

    processed_direct = r"DATA\processed"

    main(processed_direct)




