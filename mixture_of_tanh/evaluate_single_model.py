import numpy as np
import pytorch_lightning as pl
from mixture_model import MixtureModel
from kinetics_dataset import KineticsDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from plotting_utilities import cm2inch, figure_factory, sort_kinetics

def evaluate_model(model, dataloaders, figure_name: str, plotting = True):
    """Plots a figure for evaluation of a model and returns the MSE for the training and the validation data."""
    # -------------- load model --------------
    if model is str:
        my_model = MixtureModel(1)
        my_model = my_model.load_from_checkpoint(checkpoint_path=model)
    else:
        my_model = model


    # -------------- make predictions --------------
    trainer = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, logger=False)
    predictions = []
    y_values = []
    for dataloader in dataloaders:
        prediction = trainer.predict(my_model, dataloaders=dataloader)

        # concatenate batches and get true y values
        predictions.append(np.concatenate([batch.numpy() for batch in prediction]))

        # get y values
        y_value = dataloader.dataset.y
        y_values.append(y_value.numpy())

    # reshape
    predictions = [prediction.reshape((prediction.shape[0],)) for prediction in predictions]
    y_values = [y_value.reshape((y_value.shape[0],)) for y_value in y_values]

    # calculate errors
    errors = []
    y_inverse_transform = dataloaders[0].dataset.y_transform_inverse
    preds_transform = []
    ys_transform = []
    for pred_index, prediction in enumerate(predictions):
        error = np.mean((prediction-y_values[pred_index])**2)
        errors.append(error)

        # apply inverse function to retrieve true kinetic values
        preds_transform.append(y_inverse_transform(prediction))
        ys_transform.append(y_inverse_transform(y_values[pred_index]))

    # -------------- sorting --------------
    sorted_pairs = []
    for pred_index, pred in enumerate(preds_transform):
        sort_y, sort_pred = sort_kinetics(ys_transform[pred_index], pred)
        sorted_pairs.append([sort_y, sort_pred])

    # -------------- plotting --------------
    if plotting:
        # ------- create figure for train/val/test -------
        fig = figure_factory((8,6),use_params=True,yscale='log',xlabel='interfering strands',
        ylabel=r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)")

        # plot training data
        sorted_y_train, sorted_prediction_train = sorted_pairs[0]
        x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
        plt.plot(x_vals_train, sorted_y_train, color='b', label=f'training, mse={errors[0]:.2f}', linewidth=1.5)
        plt.scatter(x_vals_train, sorted_prediction_train, color='b', s=10, alpha=0.5, linewidths=0)

        # plot validation data
        sorted_y_val, sorted_prediction_val = sorted_pairs[1]
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        plt.plot(x_vals_val, sorted_y_val, color='r', label=f'validation, mse={errors[1]:.2f}', linewidth=1.5)
        plt.scatter(x_vals_val, sorted_prediction_val, color='r', s=10, alpha=0.5, linewidths=0)

        # plot test data
        sorted_y_test, sorted_prediction_test = sorted_pairs[2]
        x_vals_test = np.arange(0,1,1.0/len(sorted_y_test))
        plt.plot(x_vals_test, sorted_y_test, color='tab:olive', label=f'test, mse={errors[2]:.2f}', linewidth=1.5)
        plt.scatter(x_vals_test, sorted_prediction_test, color='tab:olive', s=10, alpha=0.5, linewidths=0)
            
        # save figure
        plt.ylim(6E-9, 2E-3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{figure_name}_train.svg", format='svg', dpi=300)
        plt.close()

        # ------- create figure for ood -------
        fig = figure_factory((8,6),use_params=True,yscale='log',xlabel='interfering strands',
        ylabel=r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)")

        # plot ood data
        sorted_y_ood, sorted_prediction_ood = sorted_pairs[3]
        x_vals_ood = np.arange(0,1,1.0/len(sorted_y_ood))
        plt.plot(x_vals_ood, sorted_y_ood, color='k', label=f'out-of-distribution, mse={errors[3]:.2f}', linewidth=1.5)
        plt.scatter(x_vals_ood, sorted_prediction_ood, color='k', s=10, alpha=0.5, linewidths=0)

        # save figure
        plt.ylim(1E-7, 2E-3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{figure_name}_ood.svg", format='svg', dpi=300)
        plt.close()

    return errors