import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits  # binary_cross_entropy
from .helper import F_score


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)                          # Generate predictions
        loss = binary_cross_entropy_with_logits(out, targets)   # Calculate loss
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)                           # Generate predictions
        loss = binary_cross_entropy_with_logits(out, targets)  # Calculate loss             # Calculate accuracy
        return {'val_outputs': torch.sigmoid(out).detach(), 'val_targets': targets.detach(), 'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        """Evaluate validation by calculating loss and f1.

        Args:
            outputs: dict with keys 'val_outputs', 'val_targets', 'val_loss'
        """
        # calculate loss
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine losses and get the mean value

        # calculate F1 score
        epoch_outputs = torch.tensor([o for batch in outputs for o in batch['val_outputs'].tolist()])
        epoch_targets = torch.tensor([o for batch in outputs for o in batch['val_targets'].tolist()])
        epoch_f1 = F_score(epoch_outputs, epoch_targets)

        return {'val_loss': epoch_loss.item(), 'val_f1': epoch_f1.item()}


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]

    if pool:
        layers.append(nn.MaxPool2d(5))  # 4
    return nn.Sequential(*layers)


class ImageClassifier(MultilabelImageClassificationBase):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # input 3 x 1280 x 1280
        self.conv1 = nn.Sequential(conv_block(in_channels, 64), nn.MaxPool2d(4))  # output 64 x 320 x 320
        self.conv2 = nn.Sequential(conv_block(64, 128), nn.MaxPool2d(4))  # output 128 x 80 x 80
        self.conv3 = nn.Sequential(conv_block(128, 512), nn.MaxPool2d(8))  # output 512 x 10 x 10
        self.classifier = nn.Sequential(nn.MaxPool2d(10),  # output 512 x 1 x 1
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512 * 1 * 1, 512),  # output 512
                                        nn.ReLU(),
                                        nn.Linear(512, num_classes))  # output 11

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out
