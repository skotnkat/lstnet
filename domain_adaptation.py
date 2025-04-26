import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from data_preparation import get_testing_loader
from torch.utils.data import DataLoader
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

    map_fn = model.map_first_to_second  # original domain is first -> mapping to second

    if orig_domain_name == model.second_domain_name:
        map_fn = model.map_second_to_first  # original domain is second -> mapping to first

    trans_dataset = translate_to_diff_domain(loader, map_fn)

    print(f'Mapped to domain with shape: {trans_dataset[0][0].shape}')

    return trans_dataset


def evaluate(clf, orig_domain_name, data_path):
    if data_path == "":
        loader = get_testing_loader(orig_domain_name)

    else:
        data = torch.load(utils.OUTPUT_FOLDER + data_path, weights_only=False)
        loader = DataLoader(data, batch_size=utils.BATCH_SIZE, shuffle=False, num_workers=utils.NUM_WORKERS)

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

        test_acc /= len(loader.dataset)

    print(f'Testing accuracy: {test_acc}')