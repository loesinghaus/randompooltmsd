from xml.dom.minidom import Element
import pytorch_lightning as pl
import torch
import torch.nn as nn
import math


class ElementWiseLinear(nn.Module):
    __constants__ = ['n_features']
    n_features: int
    weight: torch.Tensor
    
    def __init__(self, n_features: int, bias: bool = True) -> None:
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(1,n_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,n_features))
        else:
            self.register_parameter('bias', None)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.mul(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.n_features}, out_features={self.n_features}, bias={self.bias is not None}'

class MixtureModel(pl.LightningModule):
    def __init__(self, input_dims, weight_decay: float = 0):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay

        self.element_wise_linear1 = ElementWiseLinear(input_dims, bias=True)
        self.mixture_linear = nn.Linear(input_dims, input_dims, bias=True)
        self.element_wise_linear2 = ElementWiseLinear(input_dims, bias=True)
        self.output_linear = nn.Linear(input_dims, 1, bias=False)

        self.activation_function = torch.nn.functional.tanh
        self.loss_function = nn.functional.mse_loss

    def forward(self, x):
        x = self.activation_function(self.element_wise_linear1(x))
        x = self.activation_function(self.mixture_linear(x))
        x = self.activation_function(self.element_wise_linear2(x))
        x = self.output_linear(x)

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
        
        return self(x)

    def configure_optimizers(self):
        learning_rate = 0.003
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-08
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, betas = (beta_1, beta_2), eps = epsilon, weight_decay=self.weight_decay)
        # for adaptive learning rate
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [lr_scheduler]

    