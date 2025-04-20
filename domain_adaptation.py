import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from data_preparation import get_testing_loader
import utils

def translate_to_diff_domain(loader, map_fn):
    all_trans_imgs = []
    all_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(loader):
            all_labels.append(labels)

            imgs = imgs.to(utils.DEVICE)
            trans_imgs = map_fn(imgs)
            all_trans_imgs.append(trans_imgs.cpu())


    trans_igms_tensor = torch.cat(all_trans_imgs)
    labels_tensor = torch.cat(all_labels)

    return TensorDataset(trans_igms_tensor, labels_tensor)


def adapt_domain(model, orig_domain_name):
    loader = get_testing_loader(orig_domain_name)
    model.to(utils.DEVICE)
    model.eval()

    map_fn = lambda x: model.map_first_to_second(x)   # original domain is first, mapping to second
    if model.second_domain_name == orig_domain_name:
        map_fn = lambda x: model.map_second_to_first(x)  # original domain is second -> mapping to frist

    trans_dataset = translate_to_diff_domain(loader, map_fn)

    return trans_dataset
