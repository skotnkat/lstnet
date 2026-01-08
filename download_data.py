import os
import shutil
import tarfile
import urllib.request
import kagglehub


DATA_FOLDER = "data/visda2017"


def download_visda_dataset(train_op: bool = True) -> None:
    """
    Downloads and extracts the VisDA-2017 dataset.

    Args:
        train_op (bool): If True, downloads training and validation data.
                        If False, downloads test data.
    """
    VISDA_PATH = "http://csr.bu.edu/ftp/visda17/clf"

    os.makedirs(DATA_FOLDER, exist_ok=True)
    TRAIN_INPUT_PATH = VISDA_PATH + "/train.tar"
    VALIDATION_INPUT_PATH = VISDA_PATH + "/validation.tar"
    TEST_INPUT_PATH = VISDA_PATH + "/test.tar"

    # Check if extracted directories exist, if not check for tar and extract, if not download and extract
    if train_op:
        source_dir = os.path.join(DATA_FOLDER, "train")
        target_dir = os.path.join(DATA_FOLDER, "validation")
        source_tar = DATA_FOLDER + "/train.tar"
        target_tar = DATA_FOLDER + "/validation.tar"

        # Handle training data
        if not os.path.exists(source_dir):
            if os.path.exists(source_tar):
                print(
                    f"Found existing tar file. Extracting {source_tar} to {DATA_FOLDER}..."
                )
                with tarfile.open(source_tar, "r") as tar:
                    tar.extractall(DATA_FOLDER)
                os.remove(source_tar)
                print(f"Training data extracted to {source_dir} and tar file removed.")
            else:
                print(f"Downloading training data to {source_tar}...")
                _ = urllib.request.urlretrieve(TRAIN_INPUT_PATH, source_tar)
                print("Download complete.")
                print(f"Extracting {source_tar} to {DATA_FOLDER}...")
                with tarfile.open(source_tar, "r") as tar:
                    tar.extractall(DATA_FOLDER)
                os.remove(source_tar)
                print(f"Training data extracted to {source_dir} and tar file removed.")
        else:
            print(f"Training data already exists at {source_dir}")

        # Handle validation data
        if not os.path.exists(target_dir):
            if os.path.exists(target_tar):
                print(
                    f"Found existing tar file. Extracting {target_tar} to {DATA_FOLDER}..."
                )
                with tarfile.open(target_tar, "r") as tar:
                    tar.extractall(DATA_FOLDER)
                os.remove(target_tar)
                print(
                    f"Validation data extracted to {target_dir} and tar file removed."
                )
            else:
                print(f"Downloading validation data to {target_tar}...")
                _ = urllib.request.urlretrieve(VALIDATION_INPUT_PATH, target_tar)
                print("Download complete.")
                print(f"Extracting {target_tar} to {DATA_FOLDER}...")
                with tarfile.open(target_tar, "r") as tar:
                    tar.extractall(DATA_FOLDER)
                os.remove(target_tar)
                print(
                    f"Validation data extracted to {target_dir} and tar file removed."
                )
        else:
            print(f"Validation data already exists at {target_dir}")

    else:
        test_dir = DATA_FOLDER + "/visda_test"
        test_tar = DATA_FOLDER + "/test.tar"

        # Handle test data
        if not os.path.exists(test_dir):
            if os.path.exists(test_tar):
                print(
                    f"Found existing tar file. Extracting {test_tar} to {test_dir}..."
                )
                with tarfile.open(test_tar, "r") as tar:
                    tar.extractall(test_dir)
                os.remove(test_tar)
                
                # Merge all trunk folders into a single test directory
                merged_test_dir = os.path.join(test_dir, "test")
                os.makedirs(merged_test_dir, exist_ok=True)
                
                nested_test_dir = os.path.join(test_dir, "test", "test")
                if os.path.exists(nested_test_dir):
                    # First move nested test/test to test_dir level
                    for item in os.listdir(nested_test_dir):
                        src = os.path.join(nested_test_dir, item)
                        dst = os.path.join(test_dir, item)
                        shutil.move(src, dst)
                    os.rmdir(nested_test_dir)
                    os.rmdir(merged_test_dir)
                    os.makedirs(merged_test_dir, exist_ok=True)
                
                # Merge all trunk folders
                for item in os.listdir(test_dir):
                    item_path = os.path.join(test_dir, item)
                    if os.path.isdir(item_path) and item.startswith("trunk"):
                        # Move all contents from trunk folder to merged test dir
                        for file in os.listdir(item_path):
                            src = os.path.join(item_path, file)
                            dst = os.path.join(merged_test_dir, file)
                            shutil.move(src, dst)
                        os.rmdir(item_path)
                
                print(f"Test data extracted and merged to visda_test/test and tar file removed.")
            else:
                print(f"Downloading test data to {test_tar}...")
                _ = urllib.request.urlretrieve(TEST_INPUT_PATH, test_tar)
                print("Download complete.")
                print(f"Extracting {test_tar} to {test_dir}...")
                with tarfile.open(test_tar, "r") as tar:
                    tar.extractall(test_dir)
                os.remove(test_tar)
                
                # Merge all trunk folders into a single test directory
                merged_test_dir = os.path.join(test_dir, "test")
                os.makedirs(merged_test_dir, exist_ok=True)
                
                nested_test_dir = os.path.join(test_dir, "test", "test")
                if os.path.exists(nested_test_dir):
                    # First move nested test/test to test_dir level
                    for item in os.listdir(nested_test_dir):
                        src = os.path.join(nested_test_dir, item)
                        dst = os.path.join(test_dir, item)
                        shutil.move(src, dst)
                    os.rmdir(nested_test_dir)
                    os.rmdir(merged_test_dir)
                    os.makedirs(merged_test_dir, exist_ok=True)
                
                # Merge all trunk folders
                for item in os.listdir(test_dir):
                    item_path = os.path.join(test_dir, item)
                    if os.path.isdir(item_path) and item.startswith("trunk"):
                        # Move all contents from trunk folder to merged test dir
                        for file in os.listdir(item_path):
                            src = os.path.join(item_path, file)
                            dst = os.path.join(merged_test_dir, file)
                            shutil.move(src, dst)
                        os.rmdir(item_path)
                
                print(f"Test data extracted and merged to visda_test/test and tar file removed.")
        else:
            print(f"Test data already exists at {test_dir}")


def download_a2o_dataset(target_path) -> None:
    A2O_DATASET = "balraj98/apple2orange-dataset"

    if os.path.exists(target_path):
        print(f"A2O dataset already exists at {target_path}")
        return

    cache_path = kagglehub.dataset_download(A2O_DATASET)

    _ = shutil.move(cache_path, target_path)


def download_office_31_dataset(target_path) -> None:
    OFFICE_31_DATASET = "eduardolawsondasilva/office-31"

    if os.path.exists(target_path):
        print(f"Office 31 dataset already exists at {target_path}")
        return

    cache_path = kagglehub.dataset_download(OFFICE_31_DATASET)

    _ = shutil.move(cache_path, target_path)
