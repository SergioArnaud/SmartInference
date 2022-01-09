import torch
from torch import nn
from torch.nn import functional as F
import timm
from barcode_guide import barcode_guide

class OutputLayer(nn.Module):
    def __init__(
        self,
        n_digits, 
        input_neurons = 2048, 
        hidden_layers = 0, 
        dropout_rate = 0.5
    ):
        
        super().__init__()
        self.fc = FC_layers(
            input_neurons,
            dropout_rate = dropout_rate, 
            n_layers = hidden_layers
        )
        
        
        self.logits_layer = [
            nn.Linear(input_neurons, 10) for i in range(n_digits)
        ]
        self.logits_layer = nn.ModuleList(self.logits_layer)
        
    def forward(self, x):
        return [layer(x) for layer in self.logits_layer]
        
class Classifier(nn.Module):
    def __init__(self, n_classes, hidden_layers = 0, dropout_rate = 0.5, input_neurons = 2048):
        super().__init__()
        self.fc = FC_layers(
            input_neurons,
            dropout_rate = dropout_rate, 
            n_layers = hidden_layers
        )
 
        self.output = nn.Linear(input_neurons, n_classes)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.output(x)
        return x
    
class FC_layers(nn.Module):
    def __init__(self, input_neurons, dropout_rate = 0.5, n_layers = 2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = [nn.Linear(input_neurons, input_neurons) for i in range(n_layers)]
        self.hidden = nn.ModuleList(self.hidden)
        
        self.dropouts = [nn.Dropout(dropout_rate) for i in range(n_layers)]
        self.dropouts = nn.ModuleList(self.dropouts)
        
    def forward(self, x):
        if self.n_layers > 0:
            for layer, dropout in zip(self.hidden, self.dropouts):
                x = layer(x)
                x = F.relu(x)
                x = dropout(x)
        return x
    
    
        
        
        
class MC_MDCNN(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        pretrained_backbone=False,
        dropout_rate=0.1,
        n_shared_hidden = 2,
        barcode_guide=barcode_guide,
    ):
        super().__init__()
        self.barcode_guide=barcode_guide
        # Backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained_backbone, num_classes=0
        )
        self.n_neurons = self.get_n_neurons()

        # Network
        self.fc = FC_layers(
            self.n_neurons, 
            dropout_rate =  dropout_rate,
            n_layers = n_shared_hidden
        )
        
        n_classes = len(barcode_guide)
        self.classifier = Classifier(n_classes)
        
        self.outputs = [OutputLayer(barcode['n_digits'], hidden_layers=barcode['n_hidden_layers']) for barcode in self.barcode_guide]
        self.outputs = nn.ModuleList(self.outputs)
        
        # Get rainable params and initialize
        self.freeze_backbone(pretrained_backbone)
        self.initialize()
        
    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)

        # Features & FC layers
        x = self.fc(x)
            
        predicted_class = self.classifier(x)
        _, class_ = torch.max(predicted_class , 1)
        final_layer = self.outputs[torch.max(class_)]
        
        # Logits
        return final_layer(x), predicted_class
    
    def freeze_backbone(self, pretrained_backbone):
        if pretrained_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    
    def initialize(self):
        # Xavier Initialization
        for p in self.parameters():
            if p.requires_grad and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
        
        
    def get_n_neurons(self):
        try:
            return self.backbone(torch.randn(1, 3, 244, 244)).size()[1]
        except AssertionError:
            return self.backbone(torch.randn(1, 3, 384, 384)).size()[1]
        
    def random_pass(self):
        return self.forward(torch.randn(1, 3, 244, 244))


        
    
    
