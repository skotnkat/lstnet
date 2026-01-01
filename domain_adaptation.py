"""
Module is implementing img-to-img translation functions and evaluation of the translation.
Given a mapping (LSTNET model) and a dataset from one domain the model was trained on,
it translates the dataset to the other domain.
"""

from typing import Any, Callable, Optional
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation import get_testing_loader
from models.lstnet import LSTNET
import utils


def translate_to_diff_domain(
    loader: DataLoader[Any], map_fn: Callable
) -> TensorDataset:
    """Translate images from one domain to another using a mapping function.

    Args:
        loader (DataLoader[Any]): DataLoader with data to be translated.
        map_fn (Callable): Mapping function to translate the data.

    Returns:
        TensorDataset: Translated dataset (into the other domain).
    """
    all_trans_imgs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            all_labels.append(labels)

            imgs = imgs.to(utils.DEVICE)
            trans_imgs, _ = map_fn(imgs)
            all_trans_imgs.append(trans_imgs.cpu())

    trans_igms_tensor = torch.cat(all_trans_imgs)
    labels_tensor = torch.cat(all_labels)

    return TensorDataset(trans_igms_tensor, labels_tensor)


def adapt_domain(
    model: LSTNET, orig_domain_name: str, batch_size: int = 64, num_workers: int = 8, resize: Optional[int] = None
):
    """Adapt the domain of the input images using the given model.

    Args:
        model (LSTNET): The LSTNET model used for domain adaptation.
        orig_domain_name (str): The name of the original domain.
        batch_size (int, optional): The batch size for the data loader. Defaults to 64.
        num_workers (int, optional): The number of worker threads for data loading. Defaults to 8.

    Returns:
        TensorDataset: The adapted dataset.
    """
    
    loader = get_testing_loader(
        orig_domain_name,
        batch_size=batch_size,
        num_workers=num_workers,
        domain_adaptation=True,
        resize_target_size=resize,
    )

    _ = model.to(utils.DEVICE)
    _ = model.eval()

    map_fn = model.map_first_to_second  # original domain is first -> mapping to second

    if orig_domain_name == model.second_domain_name:
        map_fn = (
            model.map_second_to_first
        )  # original domain is second -> mapping to first

    trans_dataset = translate_to_diff_domain(loader, map_fn)

    print(f"Mapped to domain with shape: {trans_dataset[0][0].shape}")

    return trans_dataset


def evaluate(
    clf: Any,
    orig_domain_name: str,
    data_path: str = "",
    batch_size: int = 64,
    num_workers: int = 8,
    translated_data: Optional[TensorDataset] = None,
):
    """Evaluate the classifier on the given data.

    Args:
        clf (Any): The classifier to evaluate.
        orig_domain_name (str): The name of the original domain.
        data_path (str, optional): The path to the data file. Defaults to "".
        batch_size (int, optional): The batch size for the data loader. Defaults to 64.
        num_workers (int, optional): The number of worker threads for data loading. Defaults to 8.
        translated_data (Optional[TensorDataset], optional): The translated dataset.
            Defaults to None.

    Returns:
        float: The accuracy of the classifier on the test set.
    """
    # Use translated_data if provided, otherwise load from path or domain
    if translated_data is not None:
        loader = DataLoader(
            translated_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    elif data_path != "":
        data = torch.load(data_path, weights_only=False)
        loader = DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        loader = get_testing_loader(
            orig_domain_name, batch_size=batch_size, num_workers=num_workers
        )

    clf.to(utils.DEVICE)

    clf.eval()

    test_acc = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(utils.DEVICE)
            y = y.to(utils.DEVICE)

            outputs = clf.forward(x)
            preds = outputs.argmax(dim=1)
            acc = (preds == y).sum()
            test_acc += acc.item()

        test_acc /= len(loader.dataset)  # type: ignore

    print(f"Testing accuracy: {test_acc}")

    return test_acc
