import os
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
        target_dir = source_dir = DATA_FOLDER
        source_tar = DATA_FOLDER + "/train.tar"
        target_tar = DATA_FOLDER + "/validation.tar"

        # Handle training data
        if not os.path.exists(source_dir):
            if os.path.exists(source_tar):
                print(
                    f"Found existing tar file. Extracting {source_tar} to {source_dir}..."
                )
                with tarfile.open(source_tar, "r") as tar:
                    tar.extractall(source_dir)
                os.remove(source_tar)
                print(f"Training data extracted to visda_source and tar file removed.")
            else:
                print(f"Downloading training data to {source_tar}...")
                _ = urllib.request.urlretrieve(TRAIN_INPUT_PATH, source_tar)
                print("Download complete.")
                print(f"Extracting {source_tar} to {source_dir}...")
                with tarfile.open(source_tar, "r") as tar:
                    tar.extractall(source_dir)
                os.remove(source_tar)
                print(f"Training data extracted to visda_source and tar file removed.")
        else:
            print(f"Training data already exists at {source_dir}")

        # Handle validation data
        if not os.path.exists(target_dir):
            if os.path.exists(target_tar):
                print(
                    f"Found existing tar file. Extracting {target_tar} to {target_dir}..."
                )
                with tarfile.open(target_tar, "r") as tar:
                    tar.extractall(target_dir)
                os.remove(target_tar)
                print(
                    f"Validation data extracted to visda_target and tar file removed."
                )
            else:
                print(f"Downloading validation data to {target_tar}...")
                _ = urllib.request.urlretrieve(VALIDATION_INPUT_PATH, target_tar)
                print("Download complete.")
                print(f"Extracting {target_tar} to {target_dir}...")
                with tarfile.open(target_tar, "r") as tar:
                    tar.extractall(target_dir)
                os.remove(target_tar)
                print(
                    f"Validation data extracted to visda_target and tar file removed."
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
                print(f"Test data extracted to visda_test and tar file removed.")
            else:
                print(f"Downloading test data to {test_tar}...")
                _ = urllib.request.urlretrieve(TEST_INPUT_PATH, test_tar)
                print("Download complete.")
                print(f"Extracting {test_tar} to {test_dir}...")
                with tarfile.open(test_tar, "r") as tar:
                    tar.extractall(test_dir)
                os.remove(test_tar)
                print(f"Test data extracted to visda_test and tar file removed.")
        else:
            print(f"Test data already exists at {test_dir}")


def download_a2o_dataset() -> str:
    A2O_DATASET = "balraj98/apple2orange-dataset"
    cache_path = kagglehub.dataset_download(A2O_DATASET)

    return cache_path
