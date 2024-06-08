import torch


class train_manager(object):
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device):
        super(train_manager, self).__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer.to(device)
        self.train_loader = train_loader.to(device)
        self.val_loader = val_loader.to(device)

    def train_one_epoch(self):
        epoch_loss = 0
        i = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        # return mean loss over the whole epoch
        return epoch_loss / (i + 1)

    def train(self, epochs, eval_all_epochs=False, eval_mode=None, verbose=True):
        train_losses = []
        eval_losses = []
        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = self.train_one_epoch()
            train_losses.append(avg_loss)
            if eval_all_epochs:
                eval = self.eval_model(eval_mode=eval_mode)
                eval_losses.append(eval)
            if verbose:
                print(f'epoch{epoch}: train_loss: {avg_loss:.4f}')
                if eval_all_epochs:
                    print(f'eval_loss: {eval:.4f}')
        if eval_all_epochs:
            return train_losses, eval_losses
        else:
            return train_losses

    def eval_model(self, eval_mode=None):
        self.model.eval()
        total_loss = 0
        total_correct = 0.
        total_samples = 0.
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct = torch.sum(predicted == labels)
                total_correct += correct
                total_samples += len(labels)
        if eval_mode == 'loss':
            return total_loss / (i+1)
        elif eval_mode == 'accuracy':
            return total_correct / total_samples
        else:
            return {'loss': total_loss / (i+1), 'accuracy': total_correct / total_samples}

