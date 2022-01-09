from MC_MDCNN import MC_MDCNN

import pytorch_lightning as pl
import torch
import numpy as np
import wandb
from barcode_guide import barcode_guide

class Lit_MC_MDCNN(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet50",
        pretrained_backbone=False,
        dropout_rate=0.1,
        n_shared_hidden = 2,
        n_hidden_in_output = 0,
        lr=3e-5,
        T_0_scheduler=10,
        T_mult_scheduler=2,
        batch_size=64,
        scheduler=False,
        barcode_guide = barcode_guide,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.lr = lr
        self.batch_size = batch_size
        self.T_0_scheduler = T_0_scheduler
        self.T_mult_scheduler = T_mult_scheduler

        self.model = MC_MDCNN(
                        backbone= backbone,
                        pretrained_backbone=pretrained_backbone,
                        dropout_rate=dropout_rate,
                        n_shared_hidden=n_shared_hidden,
                        n_hidden_in_output=n_hidden_in_output,
                        barcode_guide=barcode_guide,
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y, out, log=False):
        y, real_class = y
        out, pred_class = out
        pred_class_discrete = torch.max(pred_class, 1)[1]
        n_digits = self.model.barcode_guide[pred_class_discrete]['n_digits']
        
        digits_loss = self.compute_digits_loss(y, out, log=log)
        class_loss = self.compute_class_loss(pred_class, real_class)
        loss = digits_loss + class_loss
        return loss
    
    def compute_digits_loss(self, y, out, n_digits, log=False):
        loss = 0
        if len(y) < n_digits:
            n_digits = len(y)
        for digit in range(n_digits):
            digit_loss = torch.nn.functional.cross_entropy(out[digit], y[:, digit])
            if log:
                self.log(f"digits/digit_{digit}_loss", digit_loss, on_step=False, on_epoch=True)
            loss += digit_loss
        return loss
    
    def compute_class_loss(self, y, out, log=False):
        class_loss = torch.nn.functional.cross_entropy(out, y)
        if log:
            self.log(f"Class/Class_loss", class_loss, on_step=False, on_epoch=True)
        return class_loss
        
            
    def compute_barcode_precision(self, y, out):
        y, real_class = y
        out, pred_class = out
        pred_class_discrete = torch.max(pred_class, 1)[1]
        n_digits = self.model.barcode_guide[pred_class_discrete]['n_digits']
        loss = 0
        if len(y) < n_digits:
            n_digits = len(y)

        out = torch.stack(out, 1)
        out = out[:n_digits] 
        y = y[:n_digits]
        correct = torch.sum(y == torch.max(out, 2)[1], 1)
        correct = correct.type(torch.DoubleTensor) / self.n_digits
        avg_barcode_precision = torch.mean(correct)
        self.log("val/avg_barcode_precision", avg_barcode_precision, on_step=False, on_epoch=True)
        

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        out = self.forward(x)

        loss = self.compute_loss(y, out)
        self.log("train/train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.compute_loss(y, out, log=True)
        self.compute_barcode_precision(y, out)
        #self.log_img_sample(x, y, out)
        self.log("val/val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, self.T_0_scheduler, self.T_mult_scheduler
            )
            return [optimizer], [scheduler]
        else: 
            return optimizer
