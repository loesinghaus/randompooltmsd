import numpy as np
import pytorch_lightning as pl
from MLP_model import MLP
from evaluate_single_model import evaluate_model
from kinetics_dataset import KineticsDataset
from torch.utils.data import DataLoader
import os
from torchinfo import summary
import matplotlib.pyplot as plt
from pathlib import Path
from plotting_utilities import figure_factory

# ---------------- Set number of different experiments ----------------
# the name of the experiment should match a sheet name in the input file
experiment_names = ["Choice12"]#"All", "EnsembleFull", "Ensemble", "Choice1", "Choice2",
  #"Choice3", "Choice4", "Choice5", "Choice6", "Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
# input files
path_to_excel_train = "../input_data/strand_features_both_circuits.xlsx"
path_to_excel_ood =  "../input_data/strand_features_N50.xlsx"
# train test split ratio
train_validation_test = True
split_ratio = 0.4
batch_size = 32


weight_decay_FLAG = True
if weight_decay_FLAG:
    weight_decays = [0]#,1e-4,1e-5,1e-6]
    dropout_ps = np.zeros(len(weight_decays))
else: 
    dropout_ps = [0, 0.025, 0.05, 0.1, 0.2]
    weight_decays = np.zeros(len(dropout_ps))

for experiment_name in experiment_names:
    print(f"Running experiment {experiment_name}.")

    # set the number of runs per experiment, their name, and the random seeds for the train/validation split
    run_names = [f"{experiment_name}MixedR{i*11}" for i in range(1,11)]
    random_seeds = [10*i for i in range(1,11)]
    
    # -------------- load data --------------
    
    run_train_errors = []
    run_val_errors = []
    run_test_errors = []
    run_ood_errors = []
    best_weight_decay = []

    for run_index, run_name in enumerate(run_names):
        Path(f"./experiments/{experiment_name}/{run_name}").mkdir(parents=True, exist_ok=True)

        random_seed = random_seeds[run_index]
        # create datasets
        training_dataset = KineticsDataset(path_to_excel_train, mode="train", sheet_name=experiment_name, data_split=split_ratio, random_state=random_seed)
        validation_dataset = KineticsDataset(path_to_excel_train, mode="val", sheet_name=experiment_name, data_split=split_ratio, random_state=random_seed)
        test_dataset = KineticsDataset(path_to_excel_train, mode="test", sheet_name=experiment_name, data_split=split_ratio, random_state=random_seed)
        ood_dataset = KineticsDataset(path_to_excel_ood, mode="ood", sheet_name=experiment_name, X_norm=training_dataset.X_norm[:],
            y_trans_mean=training_dataset.y_trans_mean, y_trans_std=training_dataset.y_trans_std, data_split=split_ratio, random_state=random_seed)

        # create dataloaders
        training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)
        validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
        ood_dataloader = DataLoader(ood_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
        dataloaders = [training_dataloader, validation_dataloader, test_dataloader, ood_dataloader]

        train_errors = []
        val_errors = []
        test_errors = []
        ood_errors = []
        for dropout_index, dropout_p in enumerate(dropout_ps):
            weight_decay = weight_decays[dropout_index]
        
            # -------------- create model --------------
            input_dims = len(training_dataset[0]["features"])
            my_model = MLP(input_dims=input_dims, weight_decay=weight_decay)
            #summary(my_model, input_size=(batch_size, input_dims))

            # -------------- train model --------------
            trainer = pl.Trainer(max_epochs=250, accelerator="gpu", devices=1, logger=False, checkpoint_callback=False)
            trainer.fit(my_model, train_dataloaders = training_dataloader, val_dataloaders = validation_dataloader)

            if weight_decay_FLAG:
                #figure_name = f"./experiments/{experiment_name}/{run_name}/{run_name}_d{-float(np.log10(weight_decay)):.1f}_"
                figure_name = f"./experiments/{experiment_name}/{run_name}/{run_name}_"
            else:
                figure_name = f"./experiments/{experiment_name}/{run_name}/{run_name}_p{dropout_p}_"
            errors = evaluate_model(my_model, dataloaders=dataloaders, figure_name=figure_name)
            train_errors.append(errors[0])
            val_errors.append(errors[1])
            test_errors.append(errors[2])
            ood_errors.append(errors[3])

        # write errors to file
        with open(f"./experiments/{experiment_name}/{run_name}/{run_name}_errors.txt", 'w') as f:
            f.write("weight_decay,train_error,val_error,test_error,ood_error\n")
            for weight_index, weight in enumerate(weight_decays):
                f.write(f"{weight},{train_errors[weight_index]},{val_errors[weight_index]},"
                f"{test_errors[weight_index]},{ood_errors[weight_index]}\n")

        # find best val error
        best_index = np.argmin(val_errors)
        best_weight_decay.append(best_index)
        run_train_errors.append(train_errors[best_index])
        run_val_errors.append(val_errors[best_index])
        run_test_errors.append(test_errors[best_index])
        run_ood_errors.append(ood_errors[best_index])
    
    with open(f"./errors/errors_{experiment_name}.txt", 'w') as f:
        f.write("run,weight_decay,train_error,val_error,test_error,ood_error\n")
        for run_index, run_name in enumerate(run_names):
            f.write(f"{run_name},{best_weight_decay[run_index]},{run_train_errors[run_index]},{run_val_errors[run_index]},"
            f"{run_test_errors[run_index]},{run_ood_errors[run_index]}\n")

