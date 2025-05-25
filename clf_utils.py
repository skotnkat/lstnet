import utils
from eval_models.clf_models import MnistClf, UspsClf, SvhnClf


def select_classifier(domain_name, params):
    clf = None

    if domain_name == "MNIST":
        clf = MnistClf(params)

    elif domain_name == "USPS":
        clf = UspsClf(params)

    elif domain_name == "SVHN":
        clf = SvhnClf(params)

    if clf is None:
        raise ValueError("No classifier model as loaded.")

    return clf.to(utils.DEVICE)
