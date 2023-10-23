import os
import tqdm
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from torch.optim import Adam
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50
from albumentations import (
    ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
    RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
    Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray, ToTensorV2
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# os.environ["WANDB_SILENT"] = "true"


class clr:
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'



class RSNADataset(Dataset):
    def __init__(self, dataframe, vertical_flip, horizontal_flip, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self._setup_transform()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        folder_path = row['path']
        file_list = sorted(
            os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

    def _setup_transform(self):
        if self.is_train:
            self.transform = Compose([
                Resize(height=225, width=225),
                ShiftScaleRotate(rotate_limit=90, scale_limit=[0.8, 1.2]),
                HorizontalFlip(p=self.horizontal_flip),
                VerticalFlip(p=self.vertical_flip),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(height=225, width=225),
                ToTensorV2()
            ])


class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns = no_columns
        self.output_size = output_size
        self.features = resnet50(pretrained=True)
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.classification = nn.Linear(1000 + 500, output_size)

    def forward(self, image, meta):
        image = self.features(image)
        meta = self.csv(meta)
        image_meta_data = torch.cat((image, meta), dim=1)
        out = self.classification(image_meta_data)
        return out


def load_and_prepare_data():
    train_row_data = pd.read_csv("train_dicom_row_data.csv")
    train = pd.read_csv("train.csv")
    train_df = pd.merge(
        train_row_data, train, left_on="PatientID", right_on="patient_id", how="inner")
    train_df.drop(columns=["patient_id"], inplace=True)
    a = pd.read_csv("train_dicom_row_data.csv")
    b = pd.read_csv("train.csv")
    b = b[['patient_id', 'any_injury']]
    merged_df = pd.merge(a, b, left_on='PatientID',
                         right_on='patient_id', how='inner')
    merged_df.drop(columns=["patient_id"], inplace=True)
    path = 'train_images/' + merged_df['PatientID'].astype(str) + \
        '/' + merged_df['SeriesNumber'].astype(str)
    merged_df['path'] = path
    return merged_df

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.loss_history = []
        self.accuracy_history = []

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for k, data in tqdm.tqdm(enumerate(self.dataloaders['train'])):
                image, meta, targets = data['image'].to(self.device), data['meta'].to(
                    self.device), data['target'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(image, meta)
                loss = self.criterion(outputs.squeeze(), targets.float())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * image.size(0)
                predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                correct_train += (predicted == targets).sum().item()
                total_train += targets.size(0)

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            self.loss_history.append(epoch_loss)
            train_accuracy = correct_train / total_train
            self.accuracy_history.append(train_accuracy)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            if self.scheduler:
                self.scheduler.step(epoch_loss)

    def plot_training_history(self):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def visualize_filters(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                print(f"Shape of the weights of {name}:", weights.shape)
                n_filters, n_channels = weights.shape[:2]
                n_filters_to_show = min(n_filters, 8)
                fig, axs = plt.subplots(
                    n_filters_to_show, n_channels, figsize=(15, 2 * n_filters_to_show))
                fig.suptitle(name)
                for i in range(n_filters_to_show):
                    for j in range(n_channels):
                        axs[i, j].imshow(weights[i, j])
                        axs[i, j].axis("off")
                plt.show()

if __name__ == '__main__':
    vertical_flip = 0.5
    horizontal_flip = 0.5

    selected_df = load_and_prepare_data().iloc[:100]
    ### 이부분을 바꿔주면 돌아감
    train_data, valid_data = train_test_split(selected_df, stratify=selected_df['any_injury'], test_size=0.2, shuffle=True, random_state=1234)
    train_data = RSNADataset(train_data, vertical_flip, horizontal_flip, is_train=True)
    valid_data = RSNADataset(valid_data, vertical_flip, horizontal_flip, is_train=False)

    dataloaders = {
        'train': DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2),
        'val': DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=2)
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet50Network(output_size=1, no_columns=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    trainer = ModelTrainer(model, dataloaders, criterion, optimizer, scheduler, device)
    trainer.train()
    trainer.plot_training_history()
    trainer.visualize_filters()