import pytorch_lightning as pl
import torch
import numpy as np
import wandb

from MDCNN import MDCNN


class Lit_MDCNN(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet50",
        pretrained_backbone=False,
        n_digits=13,
        variable_shape=False,
        dropout_rate=0.1,
        lr=3e-5,
        T_0_scheduler=10,
        T_mult_scheduler=2,
        batch_size=64,
        scheduler=False
    ):
        super().__init__()
        self.scheduler = scheduler
        self.lr = lr
        self.n_digits = n_digits
        self.batch_size = batch_size
        self.T_0_scheduler = T_0_scheduler
        self.T_mult_scheduler = T_mult_scheduler
        self.n_classes = 11 if variable_shape else 10

        self.model = MDCNN(
            backbone=backbone,
            pretrained_backbone=pretrained_backbone,
            n_digits=n_digits,
            variable_shape=variable_shape,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y, out, log=False):
        loss = 0
        for digit in range(self.n_digits):
            digit_loss = torch.nn.functional.cross_entropy(out[digit], y[:, digit])
            if log:
                self.log(f"digits/digit_{digit}_loss", digit_loss, on_step=False, on_epoch=True)
            loss += digit_loss

        return loss

    def log_img_sample(self, images, y, out):
        
        labels = np.random.randint(0, images.shape[0], 5)
        out = torch.stack(out).reshape(
                out[0].shape[0], self.n_classes, self.n_digits
            )
            
        for label in labels:
        
            im = images[label].T.cpu().numpy()

            vals = torch.max(out,1)[1][label].cpu().tolist()
            vals = [str(i) for i in vals]
            vals = ''.join(vals)

            vals2 = y[label].cpu().tolist()
            vals2 = [str(i) for i in vals2]
            vals2 = ''.join(vals2)

            self.log({"img": wandb.Image(im, caption=f"{vals} --- {vals2}")})
        
        
    def compute_barcode_precision(self, y, out):

        out = torch.stack(out, 1)
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