"""
Module is providing a various utility functions and global variables
for LSTNet model and its training and evaluation.
"""

from typing import Dict, Any, List, Tuple, Union, Optional, TypeAlias
import json
import torch
from torch import Tensor
from lion_pytorch import Lion

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
            model_params, lr, betas=betas, weight_decay=weight_decay
        )

    elif optim_name == "AdamW":
        optim = torch.optim.AdamW(
            model_params, lr, betas=betas, weight_decay=weight_decay
        )

    elif optim_name == "Lion":
        optim = Lion(model_params, lr, betas=betas, weight_decay=weight_decay)

    elif optim_name == "SGD":
        optim = torch.optim.SGD(
            model_params, lr, momentum=betas[0], weight_decay=weight_decay
        )

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


def convert_tensor_tuple_to_floats(
    tuple_tensor: Tuple[Tensor, ...],
) -> Tuple[float, ...]:
    """Convert a tuple of tensors to a tuple of floats."""

    return tuple(t.item() for t in tuple_tensor)

def print_gpu_memory(prefix: str = ""):
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available.")
        return

    d = torch.device("cuda")
    torch.cuda.synchronize(d)

    alloc = torch.cuda.memory_allocated(d) / 1024**2
    reserv = torch.cuda.memory_reserved(d) / 1024**2
    total = torch.cuda.get_device_properties(d).total_memory / 1024**2
    free = total - reserv

    print(
        f"[{prefix}]: alloc={alloc:.1f}MB | reserv={reserv:.1f}MB | free~={free:.1f}MB | total={total:.1f}MB"
    )

