import torch
from collections import defaultdict


def train(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, weight_decay=None):

    history = defaultdict(list)
    if weight_decay is None:
      optimizer = opt_func(model.parameters(), lr)
    else:
      optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = torch.stack(train_losses).mean().item()
        # Validation phase
        result = evaluate(model, val_loader)
        print('epoch: ', epoch, 'train_loss: ', train_loss, ' result: ', result)

        history['train_loss'].append(train_loss)
        history['val_f1'].append(result['val_f1'])
        history['val_loss'].append(result['val_loss'])

    torch.save(model, 'training/image/image.model')
    return history


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    results = [model.validation_step(batch) for batch in val_loader]
    # {'val_outputs': sigmoid(out).detach(), 'val_targets': targets.detach(), 'val_loss': loss.detach()}
    return model.validation_epoch_end(results)


def predict(model, test_dl):
    model.eval()
    results = [model.validation_step(batch) for batch in test_dl]
    outputs = [o for batch in results for o in batch['val_outputs'].tolist()]
    return outputs
