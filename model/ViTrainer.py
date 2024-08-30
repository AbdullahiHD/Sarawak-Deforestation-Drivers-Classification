import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import lr_scheduler, AdamW
import copy
from collections import defaultdict
import time
import numpy as np


class VisionTransformerMLP(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        num_classes=4,
        dropout=0.1,
        img_size=160,
        dataset_size=632,
    ):
        super(VisionTransformerMLP, self).__init__()

        self.vit = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(img_size * img_size, 512)
        self.linear2 = nn.Linear(512, 512)
        self.batch1d = nn.BatchNorm1d(512)
        self.batch1d_n = nn.BatchNorm1d(img_size * img_size)
        self.dropout = nn.Dropout(dropout)

        self.projection_head = nn.Sequential(
            nn.Linear(self.vit.num_features, 512), nn.ReLU(), nn.Linear(512, 128)
        )

        self.classifier = nn.Linear(self.vit.num_features + 512, num_classes)

    def forward(self, x, s=None, return_features=False):
        x = self.vit(x)

        if s is not None and s.numel() > 0:
            s = self.flatten(s)
            s = self.batch1d_n(s)
            s = self.dropout(s)
            s = self.batch1d(F.leaky_relu(self.linear1(s)))
            s = F.leaky_relu(self.linear2(s))
        else:
            s = torch.zeros(x.size(0), 512, device=x.device)

        combined = torch.cat((x, s), dim=1)

        if return_features:
            return combined

        output = self.classifier(combined)
        return output

    def get_projection(self, x):
        features = self.vit(x)
        return self.projection_head(features)

def train_representation(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    device,
    # patience=5,
):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = defaultdict(list)
    counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_samples = 0

            for inputs, slope, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    features = model.get_projection(inputs)
                    loss = criterion(features, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_samples += inputs.size(0)

            epoch_loss = running_loss / running_samples
            history[f"{phase}_loss"].append(epoch_loss)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                    
                else:
                    counter += 1

    time_elapsed = time.time() - start
    print(
        f"Training complete in {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best Loss: {best_loss}")

    model.load_state_dict(best_model_wts)
    return model, history

def train_classifier(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    device,
    # patience=5,
    save_path="best_clf_model.pth",
):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = defaultdict(list)
    counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for inputs, slope, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                slope = slope.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs, slope)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_samples += inputs.size(0)

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects.double() / running_samples
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                    torch.save(best_model_wts, save_path)
                    print(f"Saved best model to {save_path}")
                else:
                    counter += 1

        # if counter >= patience:
        #     print(f"Early stopping after {epoch} epochs")
        #     break

    time_elapsed = time.time() - start
    print(
        f"Training complete in {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best Loss: {best_loss}")

    model.load_state_dict(best_model_wts)
    return model, history


