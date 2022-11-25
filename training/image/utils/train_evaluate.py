
# basics
from collections import defaultdict

# ml
import torch


def train(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, weight_decay=None):
    """Train Image Classifier."""
    # initialize dict for documentations
    history = defaultdict(list)

    # initialize optimizer with or without weight_decay
    if weight_decay is None:
      optimizer = opt_func(model.parameters(), lr)
    else:
      optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)

    # for each epoch train and evaluate model
    for epoch in range(epochs):

        # model to train mode
        model.train()

        # initialize list for lossen
        train_losses = []

        # loop through batches
        for batch in train_loader:
            # forward pass
            loss = model.training_step(batch)

            # save loss
            train_losses.append(loss)

            # backwards pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # stack total loss
        train_loss = torch.stack(train_losses).mean().item()

        # validation phase
        result = evaluate(model, val_loader)
        print('epoch: ', epoch, 'train_loss: ', train_loss, ' result: ', result)

        # save losses and f1 for documentation
        history['train_loss'].append(train_loss)
        history['val_f1'].append(result['val_f1'])
        history['val_loss'].append(result['val_loss'])

    # save model after training
    torch.save(model, 'training/image/image.model')

    return history


@torch.no_grad()
def evaluate(model, val_loader):
    """Evaluate Epoch."""
    # model to validation mode
    model.eval()

    # inilialize list for results to be appended
    results = []

    # get results for each batch
    for batch in val_loader:

        # get outputs, targets and loss
        result = model.validation_step(batch)

        # add results to list
        results.append(result)

    # calculate Loss and F1 Score
    return model.validation_epoch_end(results)


def predict(model, test_dl):
    """Predict Images."""
    # model to validation mode
    model.eval()

    # inilialize list for results to be appended
    results = []

    # get results for each batch
    for batch in test_dl:

        # get outputs, targets and loss
        result = model.validation_step(batch)

        # add results to list
        results.append(result)

    # flatten outputs
    outputs = [o for batch in results for o in batch['val_outputs'].tolist()]

    return outputs
