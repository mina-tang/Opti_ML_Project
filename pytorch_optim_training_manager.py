import torch


class train_manager(object):
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device):
        super(train_manager, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_one_epoch(self, is_LSTM=False):
        epoch_loss = 0
        i = 0
        if not is_LSTM:
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        else:
            # mode for LSTM due to how labels are made
            for i, data in enumerate(self.train_loader):
                inputs = data.to(self.device)
                # learn to predict next token, label is next elem
                labels = inputs[:, 1:].to(self.device)
                outputs = self.model(inputs)
                outputs = outputs[:, :-1, :].permute(0, 2, 1)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        # return mean loss over the whole epoch
        return epoch_loss / (i + 1)

    def train(self, epochs, eval_all_epochs=False, eval_mode=None, verbose=True, is_LSTM=False, is_wine=False):
        train_losses = []
        eval_losses = []
        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = self.train_one_epoch(is_LSTM=is_LSTM)
            train_losses.append(avg_loss)
            if eval_all_epochs:
                eval = self.eval_model(eval_mode=eval_mode, is_LSTM=is_LSTM, is_wine=is_wine)
                eval_losses.append(eval)
            if verbose:
                print(f'epoch{epoch}: train_loss: {avg_loss:.4f}')
                if eval_all_epochs:
                    print(f'eval_loss: {eval:.4f}')
        if eval_all_epochs:
            return train_losses, eval_losses
        else:
            return train_losses

    def eval_model(self, eval_mode=None, is_LSTM=False, is_wine=False):
        self.model.eval()
        total_loss = 0
        total_correct = 0.
        total_samples = 0.
        if not is_LSTM:
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    total_loss += loss.item()
                    predicted = torch.argmax(outputs, dim=1)
                    if is_wine:
                        labels = torch.argmax(labels, dim=1)
                        correct = torch.sum(predicted == labels)
                    else:
                        correct = torch.sum(predicted == labels)
                    total_correct += correct
                    total_samples += len(labels)
        else:
            # mode for LSTM
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs = data.to(self.device)
                    labels = inputs[:, 1:].to(self.device)
                    outputs = self.model(inputs)
                    outputs = outputs[:, :-1, :].permute(0, 2, 1)
                    loss = self.loss_fn(outputs, labels)
                    total_loss += loss.item()
                    if type(outputs) is tuple:
                        outputs = outputs[0]
                    predicted = torch.argmax(outputs, dim=1)
                    correct = torch.sum(predicted == labels)
                    total_correct += correct
                    total_samples += labels.numel()
        if eval_mode == 'loss':
            return total_loss / (i + 1)
        elif eval_mode == 'accuracy':
            return total_correct / total_samples
        else:
            return {'loss': total_loss / (i + 1), 'accuracy': total_correct / total_samples}
