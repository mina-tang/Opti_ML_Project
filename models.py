import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.fc1 = nn.Linear(11, 10).to(dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1).to(dtype=torch.float64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN_Simple(nn.Module):
    def __init__(self):
        super(CNN_Simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=8, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, var):
        eps = torch.randn_like(var).to(self.device)
        return mean + var * eps

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x = self.decode(z)
        return x, mean, logvar


class All_CNN_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop0 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2)
        self.dp = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=0)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv9 = nn.Conv2d(192, 100, kernel_size=1)
        self.avg = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.drop0(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dp(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dp(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.avg(x)
        x = torch.squeeze(x)
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, dropout_rate, embedding_weights=None,
                 freeze_embeddings=False):
        super().__init__()
        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output
