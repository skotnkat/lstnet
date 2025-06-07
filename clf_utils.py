import utils
from eval_models.clf_models import BaseClf, SVHNCustomClf


def select_classifier(domain_name, input_size, in_channels, params, leaky_relu, custom_clf=False):
    clf = None

    if not custom_clf:
        clf = BaseClf(input_size, in_channels, params, negative_slope=leaky_relu)

    elif domain_name == "SVHN":
        clf = SVHNCustomClf()

    if clf is None:
        raise ValueError("No classifier model as loaded.")

    return clf.to(utils.DEVICE)
