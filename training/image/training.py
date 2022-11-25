"""Train Image Classifier."""

# locals
from utils.dataset import ImageDataset, DeviceDataLoader
from utils.model import ImageClassifier
from utils.train_evaluate import train
from utils.helper import get_default_device, to_device, classes

# basics
import json

# ml
from torch.utils.data import DataLoader
import torch


# choose Epochs and learning rate
EPOCHS = 60  # 80
LEARNING_RATE = 0.1


if __name__ == '__main__':

    # set maximum batch size
    batch_size = 5

    # create Datasets
    train_set = ImageDataset(file_name_annotations='image_train_labels', folder_name_pdfs='image_train', transform=True)
    val_set = ImageDataset(file_name_annotations='image_test_labels', folder_name_pdfs='image_test', transform=True)

    # create Dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # check if GPU available
    device = get_default_device()
    print('device: ', device)

    # Dataloader to GPU
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)

    # Initialize model to GPU
    model = ImageClassifier(3, len(classes))
    model = to_device(model, device)

    # define optimizer and weight_decay
    opt_func = torch.optim.SGD
    weight_decay = 1e-4

    # Train Classifier
    history = train(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        opt_func=opt_func,
        weight_decay=weight_decay
    )

    # save training history
    with open('data/image_history.json', 'r+') as f:
        json_data = json.load(f)
        json_data['cnn'] = dict(history)
        f.seek(0)
        json.dump(json_data, f, indent=4)
