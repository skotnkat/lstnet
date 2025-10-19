"""
Module is providing a various utility functions and global variables
for LSTNet model and its training and evaluation.
"""

from typing import Dict, Any, List, Tuple, Union, Optional, TypeAlias
import json
import torch
from torch import Tensor

# from lion_pytorch import Lion

PARAMS_FILE_PATH = None
NETWORK_NAMES = {
    "first_encoder",
    "second_encoder",
    "shared_encoder",
    "first_generator",
    "second_generator",
    "shared_generator",
    "first_discriminator",
    "second_discriminator",
    "latent_discriminator",
}


LOSS_LOGS: Dict[str, Dict[str, Dict[str, Any]]] = dict()

DEVICE: Optional[torch.device] = None


# requires python >=3.11
TensorPair: TypeAlias = Tuple[Tensor, Tensor]
TensorTriplet: TypeAlias = Tuple[Tensor, Tensor, Tensor]
TensorQuad: TypeAlias = Tuple[Tensor, Tensor, Tensor, Tensor]

FloatPair: TypeAlias = Tuple[float, float]
FloatTriplet: TypeAlias = Tuple[float, float, float]
FloatQuad: TypeAlias = Tuple[float, float, float, float]


def get_networks_params(params_file_path: str) -> Dict[str, Dict[str, Any]]:
    """Get the network parameters from a JSON file.

    Args:
        params_file_path (str): Path to the JSON file containing network parameters.

    Raises:
        ValueError: If any network component is missing in the JSON file.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the network parameters.
    """
    with open(params_file_path, "r", encoding="utf-8") as file:
        params = json.load(file)

    missing_network_params = NETWORK_NAMES.difference(set(params.keys()))
    if len(missing_network_params):
        raise ValueError(
            f"Missing Input Parameters for Network Components: {missing_network_params}"
        )

    return params


def transform_int_to_tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Transform an integer to a tuple of two integers."""
    if isinstance(x, int):
        x = (x, x)

    return x


def standardize_padding(
    padding: Union[int, str, Tuple[int, int]],
) -> Tuple[Tuple[int, int], bool]:
    """Standardize padding for convolutional layers.

    Args:
        padding (Union[int, str, Tuple[int, int]]): Padding configuration.

    Returns:
        Tuple[Tuple[int, int], bool]:
            A tuple containing the standardized padding and a boolean indicating if the padding is 'same'.
    """

    is_padding_same = padding == "same"
    if not isinstance(padding, str):
        padding = transform_int_to_tuple(padding)

    if not isinstance(padding, tuple):
        padding = (0, 0)

    return padding, is_padding_same


def split_padding(p_total: int) -> Tuple[int, int]:
    """Split total padding into two parts for convolutional layers."""

    p_first = p_total // 2
    p_second = p_total - p_first

    return p_first, p_second


# Will not be working with 2d data (tuples)?
def compute_effective_kernel_size(kernel_size: int, dilation: int) -> int:
    """Compute the effective kernel size for convolutional layers."""

    return dilation * (kernel_size - 1) + 1


def init_logs(ops: Optional[List[str]] = None) -> None:
    """Initialize logs for training and validation operations."""

    if ops is None:
        ops = ["train", "val"]

    for op in ops:

        LOSS_LOGS[op] = {
            "disc_loss": {"first_loss": [], "second_loss": [], "latent_loss": []},
            "enc_gen_loss": {"first_loss": [], "second_loss": [], "latent_loss": []},
            "cc_loss": {
                "first_cycle_loss": [],
                "second_cycle_loss": [],
                "first_full_cycle_loss": [],
                "second_full_cycle_loss": [],
            },
        }


def init_epoch_loss(op: str = "train"):
    """
    Initialize epoch loss for a given operation.
    The logs need to be initialized first through `init_logs`
    """
    op_logs = LOSS_LOGS[op]

    op_logs["disc_loss"]["first_loss"].append(0)
    op_logs["disc_loss"]["second_loss"].append(0)
    op_logs["disc_loss"]["latent_loss"].append(0)

    op_logs["enc_gen_loss"]["first_loss"].append(0)
    op_logs["enc_gen_loss"]["second_loss"].append(0)
    op_logs["enc_gen_loss"]["latent_loss"].append(0)

    op_logs["cc_loss"]["first_cycle_loss"].append(0)
    op_logs["cc_loss"]["second_cycle_loss"].append(0)
    op_logs["cc_loss"]["first_full_cycle_loss"].append(0)
    op_logs["cc_loss"]["second_full_cycle_loss"].append(0)


def make_loss_json_serializable(loss_val):
    if isinstance(loss_val, torch.Tensor):
        return loss_val.item()

    return loss_val


def log_epoch_loss(
    disc_loss: FloatTriplet, enc_gen_loss: FloatTriplet, cc_loss: FloatQuad, op: str
):
    """
    Log epoch loss for a given operation.

    Args:
        disc_loss (FloatTriplet): Discriminator loss.
            Consist of first, second and latent loss.
        enc_gen_loss (FloatTriplet): Encoder-Generator loss.
            Consists of first, second and latent loss.
        cc_loss (FloatQuad): Cycle consistency loss.
        first_cycle, second_cycle, first_full_cycle, second_full_cycle).
            Consists of first_cycle, second_cycle, first_full_cycle, second_full_cycle.
        op (str): Operation type.
    """

    op_logs = LOSS_LOGS[op]
    cur_epoch = len(op_logs["disc_loss"]["first_loss"]) - 1  # last epoch

    op_logs["disc_loss"]["first_loss"][cur_epoch] += make_loss_json_serializable(
        disc_loss[0]
    )
    op_logs["disc_loss"]["second_loss"][cur_epoch] += make_loss_json_serializable(
        disc_loss[1]
    )
    op_logs["disc_loss"]["latent_loss"][cur_epoch] += make_loss_json_serializable(
        disc_loss[2]
    )

    op_logs["enc_gen_loss"]["first_loss"][cur_epoch] += make_loss_json_serializable(
        enc_gen_loss[0]
    )
    op_logs["enc_gen_loss"]["second_loss"][cur_epoch] += make_loss_json_serializable(
        enc_gen_loss[1]
    )
    op_logs["enc_gen_loss"]["latent_loss"][cur_epoch] += make_loss_json_serializable(
        enc_gen_loss[2]
    )

    op_logs["cc_loss"]["first_cycle_loss"][cur_epoch] += make_loss_json_serializable(
        cc_loss[0]
    )
    op_logs["cc_loss"]["second_cycle_loss"][cur_epoch] += make_loss_json_serializable(
        cc_loss[1]
    )
    op_logs["cc_loss"]["first_full_cycle_loss"][
        cur_epoch
    ] += make_loss_json_serializable(cc_loss[2])
    op_logs["cc_loss"]["second_full_cycle_loss"][
        cur_epoch
    ] += make_loss_json_serializable(cc_loss[3])


def normalize_epoch_loss(scale, op):
    """
    Normalize epoch loss for a given operation.

    Args:
        scale (float): Scaling factor.
        op (str): Operation type.
    """

    op_logs = LOSS_LOGS[op]
    cur_epoch = len(op_logs["disc_loss"]["first_loss"]) - 1  # last epoch

    op_logs["disc_loss"]["first_loss"][cur_epoch] /= scale
    op_logs["disc_loss"]["second_loss"][cur_epoch] /= scale
    op_logs["disc_loss"]["latent_loss"][cur_epoch] /= scale

    op_logs["enc_gen_loss"]["first_loss"][cur_epoch] /= scale
    op_logs["enc_gen_loss"]["second_loss"][cur_epoch] /= scale
    op_logs["enc_gen_loss"]["latent_loss"][cur_epoch] /= scale

    op_logs["cc_loss"]["first_cycle_loss"][cur_epoch] /= scale
    op_logs["cc_loss"]["first_full_cycle_loss"][cur_epoch] /= scale
    op_logs["cc_loss"]["second_cycle_loss"][cur_epoch] /= scale
    op_logs["cc_loss"]["second_full_cycle_loss"][cur_epoch] /= scale


def check_file_ending(file, ending):
    """Check if a file has expected ending, if not, append it."""
    if not file.endswith(ending):
        file = file + ending

    return file


def init_optimizer(
    optim_name: str,
    model_params: List[torch.nn.Parameter],
    *,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Initialize optimizer.

    Args:
        optim_name (str): Name of the optimizer.
        model_params (List[torch.nn.Parameter]): Model parameters to optimize.
        lr (float): Learning rate.
        betas (Tuple[float, float]): Coefficients used for computing
            running averages of gradient and its square.
        weight_decay (float): Weight decay (L2 penalty).

    Raises:
        NotImplementedError: If the optimizer is not implemented.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """

    optim = None
    if optim_name == "Adam":
        optim = torch.optim.Adam(
            model_params, lr, betas=betas, weight_decay=weight_decay, amsgrad=True
        )

    elif optim_name == "AdamW":
        optim = torch.optim.AdamW(
            model_params, lr, betas=betas, weight_decay=weight_decay, amsgrad=True
        )

    # elif optim_name == "Lion":
    #     optim = Lion(model_params, lr, betas=betas, weight_decay=weight_decay)

    else:
        err_msg = f"Given optimizer name {optim_name} is not internally implemented yet"
        print(f"Error message: {err_msg}")

        raise NotImplementedError(err_msg)

    return optim


def get_device():
    """Get the available device (GPU if available, else CPU)."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_device():
    """Initialize the global DEVICE variable with available device (GPU if available, else CPU)."""

    global DEVICE

    if DEVICE is None:
        DEVICE = get_device()

    print(f"Using device: {DEVICE}")
