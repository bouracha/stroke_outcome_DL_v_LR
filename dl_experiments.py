import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.layers import NeuralNetworkBlock

import models.utils as model_utils
import time
from datetime import datetime

class ModelDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(np.array(x)).float()
        self.y = torch.from_numpy(np.array(y)).float()
        self.length = self.x.shape[0]
        m, n = x.shape
        self.y = self.y.unsqueeze(1)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class Classifier(nn.Module):
    def __init__(self, layers=[1024, 512, 128, 32, 16], batch_norm=True, p_dropout=0.0, device="cuda",
                 tensorboard=False):
        super(Classifier, self).__init__()

        self.device = device
        self.tensorboard = tensorboard
        self.net = NeuralNetworkBlock(layers=layers, activation=nn.LeakyReLU(0.1), batch_norm=batch_norm,
                                      p_dropout=p_dropout)

        self.accum_loss = dict()

    def forward(self, x):
        x = torch.sigmoid(self.net(x))
        return x

    def cal_loss(self, y, y_hat):
        loss_fn = nn.BCELoss()
        loss = loss_fn(y_hat, y)
        return loss

    def cal_metrics(self, y, y_hat):
        y_hat = np.where(y_hat > 0.5, 1, 0)

        acc = accuracy_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        recall = recall_score(y, y_hat)

        fpr, tpr, thresholds = roc_curve(y, y_hat)
        AUC = auc(fpr, tpr)
        return AUC

    def train_step(self, train_loader):

        for i, (X, y) in enumerate(tqdm(train_loader, disable=True)):
            cur_batch_size = len(X)

            inputs = X.to(self.device).float()

            y_hat = self(inputs.float())

            y = y.to(self.device).float()
            loss = self.cal_loss(y, y_hat)
            # AUC = self.cal_metrics(y, y_hat)

            self.optimizer.zero_grad()
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            model_utils.accum_update(self, 'train_loss', loss)

        if self.tensorboard:
            self.writer.add_scalar("losses/train_loss", self.accum_loss['train_loss'].avg, self.epoch)

    def eval_step(self, val_loader, test_loader):
        with torch.no_grad():
            self.eval()
            for i, (X, y) in enumerate(tqdm(val_loader, disable=True)):
                cur_batch_size = len(X)

                inputs = X.to(self.device).float()

                y_hat = self(inputs.float())

                y = y.to(self.device).float()
                loss = self.cal_loss(y, y_hat)
                AUC = self.cal_metrics(y.cpu(), y_hat.cpu())

                model_utils.accum_update(self, 'val_loss', loss)
                model_utils.accum_update(self, 'val_auc', AUC)
            for i, (X, y) in enumerate(tqdm(test_loader, disable=True)):
                cur_batch_size = len(X)

                inputs = X.to(self.device).float()

                y_hat = self(inputs.float())

                y = y.to(self.device).float()
                loss = self.cal_loss(y, y_hat)
                AUC = self.cal_metrics(y.cpu(), y_hat.cpu())

                model_utils.accum_update(self, 'test_loss', loss)
                model_utils.accum_update(self, 'test_auc', AUC)

            if self.tensorboard:
                self.writer.add_scalar("losses/val_loss", self.accum_loss['val_loss'].avg, self.epoch)
                self.writer.add_scalar("losses/val_auc", self.accum_loss['val_auc'].avg, self.epoch)

            # if self.accum_loss['val_auc'].avg > self.val_auc_max_val:
            if self.accum_loss['val_loss'].avg < self.val_error_min_val:
                self.val_error_min_val = self.accum_loss['val_loss'].avg
                self.val_auc_max_val = self.accum_loss['val_auc'].avg
                self.test_error_min_val = self.accum_loss['test_loss'].avg
                self.test_auc_max_val = self.accum_loss['test_auc'].avg





def dl_experiment(X_train, X_val, X_test, y_train, y_val, y_test, num_epochs=10, model_lr=0.01, p_dropout=0.0,
               model_BatchNorm=True, weight_decay=1e-4, device="cpu", tensorboard=True):
    input_n = X_train.shape[1]

    # layers=[input_n, 1]
    layers=[input_n, 256, 1]
    #layers = [input_n, 512, 256, 128, 64, 32, 16, 8, 1]
    model = Classifier(layers=layers, batch_norm=model_BatchNorm, p_dropout=p_dropout, device=device,
                       tensorboard=tensorboard)
    # print(model)
    model.val_error_min_val = 10000
    model.val_auc_max_val = 0.5
    model.test_error_min_val = 10000
    model.test_auc_max_val = 0.5
    model.to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=model_lr, weight_decay=weight_decay)
    if model.tensorboard:
        model.writer = SummaryWriter()

    train_dataset = ModelDataset(X_train, y_train)
    val_dataset = ModelDataset(X_val, y_val)
    test_dataset = ModelDataset(X_test, y_test)

    batch_size = 100
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=999999, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=999999, shuffle=True)

    num_epochs = num_epochs
    for model.epoch in range(0, num_epochs):
        model.train_step(train_loader)
        model.eval_step(val_loader, test_loader)

    if model.tensorboard:
        model.writer.close()

    return model.val_error_min_val, model.val_auc_max_val, model.test_error_min_val, model.test_auc_max_val

def run_dl_experiment(data, labels, train_sizes, num_runs=5, device="cuda", tensorboard=True, BatchNorm=True, p_dropout=0.0, lr=0.01, weight_decay=1e-4, num_samples=float(3e5)):
    final_results = []

    for train_size in train_sizes:
        num_epochs = int((1.0*num_samples) / (1.0 * train_size))
        val_error_min = []
        val_auc_max = []
        test_error_min = []
        test_auc_max = []
        for i in range(0, num_runs):
            print('Run number: %i/%i' % (i + 1, num_runs), end='\r')

            ### Make data split
            # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=random.seed(datetime.now()))
            X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels,
                                                                                        test_size=0.20, random_state=i,
                                                                                        stratify=labels)  # random.seed(datetime.now()))
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                                              test_size=0.20, random_state=i,
                                                                              stratify=y_train_val)  # random.seed(datetime.now()))

            X_train = X_train[:train_size]
            y_train = y_train[:train_size]

            best_val_auc = 0.5
            for lr in [0.1, 0.03, 0.01, 0.003, 0.001]:
                val_error_min_val, val_auc_max_val, test_error_min_val, test_auc_max_val = dl_experiment(X_train,
                                                                                                     X_val,
                                                                                                     X_test,
                                                                                                     y_train, y_val,
                                                                                                     y_test,
                                                                                                     num_epochs=num_epochs,
                                                                                                     model_lr=lr,
                                                                                                     p_dropout=p_dropout,
                                                                                                     model_BatchNorm=BatchNorm,
                                                                                                     weight_decay=weight_decay,
                                                                                                     device=device,
                                                                                                     tensorboard=tensorboard)
                if val_auc_max_val >= best_val_auc:
                    best_lr = lr
                    best_weight_decay = weight_decay
                    best_val_auc = val_auc_max_val
                    best_val_auc_val_entropy = val_error_min_val
                    best_val_auc_test_entropy = test_error_min_val
                    best_val_auc_test_auc = test_auc_max_val


            #print("lr/weight_decay selected: ", best_lr, best_weight_decay)
            val_error_min.append(best_val_auc_val_entropy)
            val_auc_max.append(best_val_auc)
            test_error_min.append(best_val_auc_test_entropy)
            test_auc_max.append(best_val_auc_test_auc)


        print("#############################################################################################")
        print("For train size {}, Val BCE = {:.4f},{:.4f}, Val AUC = {:.4f},{:.4f}".format(train_size,
                                                                                           np.mean(val_error_min),
                                                                                           np.std(val_error_min),
                                                                                           np.mean(val_auc_max),
                                                                                           np.std(val_auc_max)))
        print("For train size {}, Test BCE = {:.4f},{:.4f}, Test AUC = {:.4f},{:.4f}".format(train_size,
                                                                                             np.mean(test_error_min),
                                                                                             np.std(test_error_min),
                                                                                             np.mean(test_auc_max),
                                                                                             np.std(test_auc_max)))

        final_results.append("For train size {}, Val BCE = {:.4f},{:.4f}, Val AUC = {:.4f},{:.4f}".format(train_size,
                                                                                                          np.mean(
                                                                                                              val_error_min),
                                                                                                          np.std(
                                                                                                              val_error_min),
                                                                                                          np.mean(
                                                                                                              val_auc_max),
                                                                                                          np.std(
                                                                                                              val_auc_max)))
        final_results.append("For train size {}, Test BCE = {:.4f},{:.4f}, Test AUC = {:.4f},{:.4f}".format(train_size,
                                                                                                            np.mean(
                                                                                                                test_error_min),
                                                                                                            np.std(
                                                                                                                test_error_min),
                                                                                                            np.mean(
                                                                                                                test_auc_max),
                                                                                                            np.std(
                                                                                                                test_auc_max)))

    return final_results
