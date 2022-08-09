from tkinter import Y
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split

class KineticsDataset(Dataset):
    def __init__(self, xlsx_file_path: str, sheet_name = "None", mode = "train", X_norm = None,
         y_trans_mean = None, y_trans_std = None, data_split=0.4, random_state=11):
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)

        # Define X and y (manual dropping of questions here)
        X = df.drop("kinetics", axis=1).to_numpy()
        
        # create X normalization function
        if X_norm is None:
            self.X_norm = [0, 0]
            self.X_norm[0] = np.mean(X, axis=0, keepdims=True)
            self.X_norm[1] = np.std(X, axis=0, keepdims=True)
        else:
            self.X_norm = X_norm
        X = (X-self.X_norm[0])/(self.X_norm[1]+1e-8)

        # rescale kinetics
        if not y_trans_mean is None:
            self.y_trans_mean = y_trans_mean
            self.y_trans_std = y_trans_std

        y = df["kinetics"].to_numpy()
        self.original_y = np.copy(y)
        if mode!="ood":
            y, self.y_transform_inverse, self.y_trans_mean, self.y_trans_std = self.y_transform(y)
        else:
            y = self.y_transform_fixed(y, self.y_trans_mean, self.y_trans_std)
            
        if mode != "ood":
            # Define train/val/test split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=data_split,
            random_state=random_state, shuffle=True)
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5,
            random_state=random_state, shuffle=True)

        if mode == "train":
            self.X = torch.tensor(X_train,dtype=torch.float32)
            self.y = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
        elif mode == "val":
            self.X = torch.tensor(X_val,dtype=torch.float32)
            self.y = torch.tensor(y_val,dtype=torch.float32).unsqueeze(1)
        elif mode == "test":
            self.X = torch.tensor(X_test,dtype=torch.float32)
            self.y = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)
        elif mode == "ood":
            self.X = torch.tensor(X,dtype=torch.float32)
            self.y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)
        else:
            raise ValueError(f"{mode} is not a valid mode.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {"features": self.X[idx], "kinetics": self.y[idx]}
        return sample

    def y_inverse(self, y, mean, std):
        y = y*std+mean
        y = np.exp(y)

        return y

    def y_transform_fixed(self, y, mean, std):
        y = np.log(y)
        y = (y-mean)/std

        return y

    def y_transform(self, y):
        y = np.log(y)
        mean = np.mean(y)
        std = np.std(y)
        y = (y-mean)/std
        inverse_function = partial(self.y_inverse, mean=mean, std=std)

        return y, inverse_function, mean, std