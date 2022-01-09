import torch
from torch import nn
from torch.nn import functional as F
import timm


class MDCNN(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        pretrained_backbone=False,
        n_digits=13,
        variable_shape=False,
        dropout_rate=0.1,
    ):

        super().__init__()

        # Set parameters
        self.n_classes = 11 if variable_shape else 10
        self.n_digits = n_digits

        # Backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained_backbone, num_classes=0
        )
        self.n_neurons = self.get_n_neurons()

        # Network
        self.dense1 = nn.Linear(self.n_neurons, self.n_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(self.n_neurons, self.n_neurons)
        self.logits_layer = [
            nn.Linear(self.n_neurons, self.n_classes) for i in range(n_digits)
        ]
        self.logits_layer = nn.ModuleList(self.logits_layer)

        # Get rainable params and initialize
        self.freeze_backbone(pretrained_backbone)
        self.initialize()

    def get_n_neurons(self):
        try:
            return self.backbone(torch.randn(1, 3, 244, 244)).size()[1]
        except AssertionError:
            return self.backbone(torch.randn(1, 3, 384, 384)).size()[1]

    def freeze_backbone(self, pretrained_backbone):
        if pretrained_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def initialize(self):
        # Xavier Initialization
        for p in self.parameters():
            if p.requires_grad and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)

        # Features & FC layers
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = F.relu(x)

        # Logits
        return [layer(x) for layer in self.logits_layer]

    def random_pass(self):
        return self.forward(torch.randn(1, 3, 244, 244))