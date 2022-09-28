import pytorch_lightning as pl
import torch
import torch.nn as nn

class MLP(pl.LightningModule):
    def __init__(self, input_dims, dropout_p=0.2, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay

        self.layer_size=30

        self.linear1 = nn.Linear(input_dims, self.layer_size)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.linear3 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.linear4 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout4 = nn.Dropout(p=dropout_p)
        self.linear5 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout5 = nn.Dropout(p=dropout_p)
        self.linear6 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout6 = nn.Dropout(p=dropout_p)
        self.linear7 = nn.Linear(self.layer_size, self.layer_size)
        self.dropout7 = nn.Dropout(p=dropout_p)
        self.linear8 = nn.Linear(self.layer_size, 1)

        self.activation_function = nn.functional.relu
        self.loss_function = nn.functional.mse_loss

    def forward(self, x):
        # -------- first linear layer -----------
        x = self.activation_function(self.linear1(x))

        # -------- first block -----------
        residual_1 = x.clone()
        x = self.dropout1(x)
        x = self.activation_function(self.linear2(x))
        x = self.dropout2(x)
        # add residual
        x = self.activation_function(self.linear3(x)+residual_1)

        # -------- second block -----------
        residual_2 = x.clone()
        x = self.dropout3(x)
        x = self.activation_function(self.linear4(x))
        x = self.dropout4(x)
        # add residual
        x = self.activation_function(self.linear5(x)+residual_2)

        # -------- third block -----------
        residual_3 = x.clone()
        x = self.dropout5(x)
        x = self.activation_function(self.linear6(x))
        x = self.dropout6(x)
        # add residual
        x = self.activation_function(self.linear7(x)+residual_3)

        # -------- output linear activation -----------
        x = self.linear8(x)

        return x

    def training_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["kinetics"]

        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["kinetics"]

        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["kinetics"]
        
        return self(x)

    def configure_optimizers(self):
        learning_rate = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, betas = (beta_1, beta_2), eps = epsilon, weight_decay=self.weight_decay)
        # for adaptive learning rate
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [lr_scheduler]

    