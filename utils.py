import os
import shutil
import pandas as pd
import torch.utils.data
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose

from DeepLearnerLib.CONSTANTS import DEFAULT_FILE_PATHS
from DeepLearnerLib.data_utils.CustomDataset import GeomCnnDataset


def get_image_files_single_scalar(FILE_PATHS, data_dir="TRAIN_DATA_DIR"):
    """
    Retrieve image files and their corresponding labels for training/testing.
    
    Args:
        FILE_PATHS (dict): Dictionary containing paths for data directories and file details.
        data_dir (str): The directory containing training data (default is 'TRAIN_DATA_DIR').
    
    Returns:
        tuple: A tuple containing a list of file paths and a list of labels.
    """
    print("Using file paths:", FILE_PATHS)
    

    file_names = []
    labels = []

    if FILE_PATHS is None:
        FILE_PATHS = DEFAULT_FILE_PATHS

    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    print("Sorted subject IDs")

 
    scalars = FILE_PATHS["FEATURE_DIRS"]
    time_points = FILE_PATHS["TIME_POINTS"]

    attr = get_attributes(FILE_PATHS)
    count = {0: 0, 1: 0}


    for sub in subject_ids:
        subject_path = os.path.join(FILE_PATHS[data_dir], sub)

        if not os.path.isdir(subject_path):
            continue

        feat_tuple = []
        session_paths = [os.path.join(subject_path, t) for t in time_points]
        
        sub_attr = attr.loc[attr["CandID"] == int(sub)]

        if sub_attr.empty:
            continue
        complete_data = all(os.path.isdir(session_path) for session_path in session_paths)
        if not complete_data:
            continue

        group = int(sub_attr["group"].values[0])
        labels.append(group)
        count[group] += 1

        for session_path in session_paths:
            if not os.path.isdir(session_path):
                continue
            
            n_feat = [os.path.join(session_path, f) for f in scalars if os.path.isdir(os.path.join(session_path, f))]
            if len(n_feat) == 0:
                continue

            for scalar in scalars:
                feat_tuple.append(os.path.join(session_path, scalar, f"left_{scalar}{FILE_PATHS['FILE_SUFFIX'][0]}") + FILE_PATHS["FILE_EXT"])
                feat_tuple.append(os.path.join(session_path, scalar, f"right_{scalar}{FILE_PATHS['FILE_SUFFIX'][1]}") + FILE_PATHS["FILE_EXT"])
        
        file_names.append(feat_tuple)
    
    return file_names, labels


def get_test_dataloader():
    test_files, test_labels = get_image_files_single_scalar("TEST_DATA_DIR")
    test_transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensity(),
        EnsureType()
    ])
    
    dataset = GeomCnnDataset(test_files, test_labels, test_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=100)


def get_attributes(FILE_PATHS):

    file_path = os.path.join(FILE_PATHS["CSV_path"])
    if not os.path.exists(file_path):
        print(f"File {file_path} doesn't exist.")
        exit(1)
    return pd.read_csv(file_path)


def anonymize_dataset(FILE_PATHS):

    output_path = "/Users/mturja/Desktop/sample_dataset_anonymous"
    attr = get_attributes(FILE_PATHS)
    subject_ids = sorted(os.listdir(FILE_PATHS["TRAIN_DATA_DIR"]))
    
    attr = attr[attr[FILE_PATHS["id_name"]].isin([int(s) for s in subject_ids])]
    
    for i, sub in enumerate(subject_ids):
        sub_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], sub)
        if not os.path.isdir(sub_path):
            continue
        
        sub_attr = attr.loc[attr[FILE_PATHS["id_name"]] == int(sub)]
        if sub_attr.empty:
            continue
        
        group = sub_attr[FILE_PATHS["group_name"]].values[0]

        if "LR" in group:
            attr.drop(attr[attr[FILE_PATHS["id_name"]] == int(sub)].index, inplace=True)
            continue

        print(f"Copying data for subject {sub}...")
        

        attr.loc[attr[FILE_PATHS["id_name"]] == int(sub), [FILE_PATHS["id_name"], FILE_PATHS["group_name"]]] = [i, group]

        shutil.copytree(sub_path, os.path.join(output_path, str(i)))

    attr.to_csv(os.path.join(output_path, "output_csv.csv"))

if __name__ == '__main__':
    anonymize_dataset(DEFAULT_FILE_PATHS)
