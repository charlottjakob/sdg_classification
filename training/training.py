import torch
import torchvision
from torchvision import transforms, datasets


train = datasets.MINST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MINST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
