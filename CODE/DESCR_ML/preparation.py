import os
import pandas as pd

def main(train_data_directory, test_data_directory, save_directory):
    """ Note from the owner of the dataset: 
        "I think SARTAJ dataset has a problem that the glioma class images are not categorized 
        correctly, I realized this from the results of other people's work as well as 
        the different models I trained, which is why I deleted the images in this folder 
        and used the images on the figshare site" 
        """

    labels = ["notumor", "meningioma", "pituitary"] # As suggested in the comment above, I removed the glioma folder
    test = []
    train = []
    for label in labels:
        directory = os.path.join(train_data_directory, label)
        image_list = os.listdir(directory)
        for filename in image_list[:]:
            if filename.endswith('.jpg'):
                row = {"filename": filename, "path": os.path.join(directory, filename), "label": label}
                train.append(row)
    for label in labels:
        directory = os.path.join(test_data_directory, label)
        image_list = os.listdir(directory)
        for filename in image_list[:]:
            if filename.endswith('.jpg'):
                row = {"filename": filename, "path": os.path.join(directory, filename), "label": label}
                test.append(row)
    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)

    df_train.to_csv(os.path.join(save_directory, "train_dataset.csv"), sep = "|", index = False)
    df_test.to_csv(os.path.join(save_directory, "test_dataset.csv"), sep = "|", index = False)


if __name__ == "__main__":
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    train_data_directory = r"DATA\Training"
    test_data_directory = r"DATA\Testing"
    save_directory = r"DATA\datasets"

    main(train_data_directory, test_data_directory, save_directory)