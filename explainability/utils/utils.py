import torch

MAX_LEN = 512


# helper functions to load the data and model onto GPU
def get_default_device():
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')