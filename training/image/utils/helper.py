import torch


classes = range(1, 18)

# helper functions to load the data and model onto GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def F_score(output, label, threshold=0.5, beta=1):
    """Calculate F1-score."""
    # divide otputs to True and False
    prob = output > threshold

    # transform 0/1 to True and False
    label = label > threshold

    true_positive = (prob & label).sum(1).float()
    false_positive = (prob & (~label)).sum(1).float()
    false_negative = ((~prob) & label).sum(1).float()

    precision = torch.mean(true_positive / (true_positive + false_positive + 1e-12))
    recall = torch.mean(true_positive / (true_positive + false_negative + 1e-12))
    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return f_score.mean(0)
