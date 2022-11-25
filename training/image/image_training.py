from utils.dataset import ImageDataset, DeviceDataLoader
from utils.model import ImageClassifier
from utils.train_evaluate import train
from utils.helper import get_default_device, to_device

from torch.utils.data import DataLoader
import torch
import json

BATCH_SIZE = 5
epochs = 5  # 80
lr = 0.1
opt_func = torch.optim.SGD
weight_decay = 1e-4
classes = range(1, 18)

train_set = ImageDataset(file_name_annotations='image_train_labels', folder_name_pdfs='image_train', transform=True)
val_set = ImageDataset(file_name_annotations='image_test_labels', folder_name_pdfs='image_test', transform=True)


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

device = get_default_device()
print('device: ', device)

train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)

model = to_device(ImageClassifier(3, len(classes)), device)

history = train(epochs, lr, model, train_dl, val_dl, opt_func=opt_func, weight_decay=weight_decay)

with open('data/image_history.json', 'r+') as f:
    json_data = json.load(f)
    json_data['cnn'] = dict(history)
    f.seek(0)
    json.dump(json_data, f, indent=4)
