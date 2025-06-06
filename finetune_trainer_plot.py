import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from models.model_for_faced import Model
from tqdm import tqdm
import torch
from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import umap
from sklearn.decomposition import PCA
import copy
import os


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'I-CARE-CPC']:
            # Use class weights if available
            weight = getattr(params, 'class_weights', None)
            self.criterion = CrossEntropyLoss(weight=self.params.class_weights.cuda(), label_smoothing=self.params.label_smoothing).cuda()
            # self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'I-CARE-Outcome']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        
        # Store metrics
        metrics = {
        'epoch': [],
        'train_loss': [],
        'val_acc': [], 'val_pr_auc': [], 'val_roc_auc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [], 'val_sensitivity': [], 'val_specificity': [],
        }
        
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                
                losses.append(loss.data.cpu().numpy())
                
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            avg_train_loss = np.mean(losses)


            with torch.no_grad():
                acc, pr_auc, roc_auc, f1, precision, recall, sensitivity, specificity, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                
                # Store metrics
                metrics['epoch'].append(epoch + 1)
                metrics['train_loss'].append(avg_train_loss)
                metrics['val_acc'].append(acc)
                metrics['val_pr_auc'].append(pr_auc)
                metrics['val_roc_auc'].append(roc_auc)
                metrics['val_f1'].append(f1)
                metrics['val_precision'].append(precision)
                metrics['val_recall'].append(recall)
                metrics['val_sensitivity'].append(sensitivity)
                metrics['val_specificity'].append(specificity)
                
                print(
                    "Epoch {:>2} : Training Loss: {:.5f} |"
                    "Acc: {:.5f} | PR_AUC: {:.5f} | ROC_AUC: {:.5f} |"
                    "F1: {:.5f} | Precision: {:.5f} | Recall: {:.5f} | "
                    "Sens: {:.5f} | Spec: {:.5f} | LR: {:.5f} | "
                    "Time: {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        f1,
                        precision,
                        recall,
                        sensitivity,
                        specificity,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print("Confusion Matrix:\n", cm)
                
                if roc_auc > roc_auc_best:
                    print("ROC-AUC improved. Saving best weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    
        # Load best checkpoint
        self.model.load_state_dict(self.best_model_states)
        
        with torch.no_grad():
            print("***************************Test************************")
            print("\n" + "*" * 20 + " TEST " + "*" * 20)
            (acc,
             pr_auc,
             roc_auc,
             f1,
             precision,
             recall,
             sensitivity,
             specificity,
             cm)  = self.test_eval.get_metrics_for_binaryclass(self.model)
            
            print("***************************Test results************************")
            print("→ Final Test Metrics:")
            print(
                "Acc: {:.5f} | PR AUC: {:.5f} | ROC AUC: {:.5f} | "
                "F1: {:.5f} | Precision: {:.5f} | Recall: {:.5f} | "
                "Sens: {:.5f} | Spec: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                    f1,
                    precision,
                    recall,
                    sensitivity,
                    specificity,
                )
            )
            print("Confusion Matrix:\n", cm)
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
                
            # model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            model_path = (
                f"{self.params.model_dir}/"
                f"epoch{best_epoch:02d}_acc_{acc:.5f}_pr_{pr_auc:.5f}_"
                f"roc_{roc_auc:.5f}_f1_{f1:.5f}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            print(f"model save to {model_path}")
            
        # ↓↓↓ PLOTTING (every 5 epochs) ↓↓↓
        plot_epochs = [e for e in metrics['epoch'] if e % 5 == 0]
        
        # utility for every 5th epoch
        def downsample(key):
            return [val for i, val in enumerate(metrics[key]) if metrics['epoch'][i] % 5 == 0]
            
            
        fig, axs = plt.subplots(2, 4, figsize=(24, 10))
        axs = axs.flatten()
        metric_keys = [
        ('val_acc', 'Accuracy'),
        ('val_pr_auc', 'PR AUC'),
        ('val_roc_auc', 'ROC AUC'),
        ('val_f1', 'F1 Score'),
        ('val_precision', 'Precision'),
        ('val_recall', 'Recall'),
        ('val_sensitivity', 'Sensitivity'),
        ('val_specificity', 'Specificity'),
    ]

    for i, (key, label) in enumerate(metric_keys):
        axs[i].plot(plot_epochs, downsample(key), marker='o', linestyle='-')
        axs[i].set_title(label)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(label)
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(self.params.model_dir, "metrics_plot.png"))
    plt.close()


    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
