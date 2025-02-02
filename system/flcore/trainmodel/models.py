from collections import OrderedDict
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn


batch_size = 16

class LocalModel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(LocalModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        
    def forward(self, x, feat=False):
        out = self.feature_extractor(x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out


# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 500), 
            nn.ReLU(),
            nn.Linear(500, 100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


class CNNHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, dim, client_sample, num_classes=10, hidden_dim=100, n_hidden=1, spec_norm=False):
        super(CNNHyper, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.client_sample = client_sample
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)
        self.head_value = nn.Linear(hidden_dim, self.dim * self.num_classes)
        self.head_bias = nn.Linear(hidden_dim, self.num_classes)

    def finetune(self, emd):
        features = self.mlp(emd)
        weights = OrderedDict()
        head_value = self.head_value(features).view(self.num_classes, self.dim)
        head_bias = self.head_value(features).view(-1)
        weights["weight"] = head_value
        weights["bias"] = head_bias
        return weights


    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        if test == False:
            weights = [OrderedDict() for x in range(self.client_sample)]
            head_value = self.head_value(features).view(-1, self.num_classes, self.dim)
            head_bias = self.head_bias(features).view(-1, self.num_classes)
            for nn in range(self.client_sample):
                weights[nn]["weight"] = head_value[nn]
                weights[nn]["bias"] = head_bias[nn]
        else:
            weights = OrderedDict()
            head_value = self.head_value(features).view(self.num_classes, self.dim)
            head_bias = self.head_bias(features).view(self.num_classes)
            weights["weight"] = head_value
            weights["bias"] = head_bias

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z

        return out
