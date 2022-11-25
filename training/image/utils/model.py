import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits  # binary_cross_entropy
from .helper import F_score


class MultilabelImageClassificationBase(nn.Module):
    """Includes Training functions to be run on GPU."""

    def training_step(self, batch):
        """Forward pass and Loss Calculation of single batch."""
        # split to data and targets
        images, targets = batch

        # forward pass
        out = self(images)

        # calculate loss
        loss = binary_cross_entropy_with_logits(out, targets)   # Calculate loss

        return loss

    def validation_step(self, batch):
        """Validate single batch."""
        # split to data and targets
        images, targets = batch

        # forward pass
        out = self(images)

        # calculate loss
        loss = binary_cross_entropy_with_logits(out, targets)  # Calculate loss

        # return outputs, targets loss to calculate F1 and Loss for total epoch
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


class ImageClassifier(MultilabelImageClassificationBase):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # input 3 x 1280 x 1280
        self.conv1 = nn.Sequential(self.conv_block(in_channels, 64), nn.MaxPool2d(4))  # output 64 x 320 x 320
        self.conv2 = nn.Sequential(self.conv_block(64, 128), nn.MaxPool2d(4))  # output 128 x 80 x 80
        self.conv3 = nn.Sequential(self.conv_block(128, 512), nn.MaxPool2d(8))  # output 512 x 10 x 10
        self.classifier = nn.Sequential(nn.MaxPool2d(10),  # output 512 x 1 x 1
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512 * 1 * 1, 512),  # output 512
                                        nn.ReLU(),
                                        nn.Linear(512, num_classes))  # output 17

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out

    def conv_block(self, in_channels, out_channels):
        """Build Convolution block for Model."""

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolutional Layer
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU(inplace=True)  # ReLU
        ]

        # transfrom list to Sequence and return
        return nn.Sequential(*layers)
