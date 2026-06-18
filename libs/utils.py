import os
import random

import pywt
import torch
import math
import numpy as np

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# metric
def metric(pred, label):
    pred = torch.as_tensor(pred, dtype=torch.float32)
    label = torch.as_tensor(label, dtype=torch.float32)
    mae = masked_mae(pred, label, 0.0).item()
    rmse = masked_rmse(pred, label, 0.0).item()
    mape = masked_mape(pred, label, 0.0).item()
    return mae, rmse, mape

def compute_loss(Y, YL, YH, y_hat, y_hat_l, y_hat_h, std, mean,
                 lambda_pred=1.0, lambda_aux=0.5, lambda_rec=0.1):
    y_hat_real = y_hat * std + mean
    y_hat_l_real = y_hat_l * std + mean
    y_hat_h_real = y_hat_h * std + mean

    loss_pred = masked_mae(y_hat_real,Y, 0.0)

    loss_low = masked_mae(y_hat_l_real, YL, 0.0)
    loss_high = masked_mae(y_hat_h_real, YH, 0.0)
    loss_aux = (loss_low + loss_high) / 2.0

    loss_rec = masked_mae(y_hat_l_real + y_hat_h_real, y_hat_real,0.0)
    total_loss = (lambda_pred * loss_pred) + (lambda_aux * loss_aux) + (lambda_rec * loss_rec)

    return total_loss, loss_pred, loss_aux, loss_rec


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > 1e-5)
    mask = mask.float()
    mask  /= (torch.mean(mask) + 1e-5)

    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def masked_rmse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > 1e-5)
    mask = mask.float()
    mask /= (torch.mean(mask) + 1e-5)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.square(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))
def masked_mape(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > 1e-5)
    mask = mask.float()
    mask /= (torch.mean(mask)+1e-5)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / torch.clamp(labels,min=1e-5)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    num_sample = data.shape[0] - P - Q + 1
    x = np.stack([data[i: i + P] for i in range(num_sample)])
    y = np.stack([data[i + P: i + P + Q] for i in range(num_sample)])
    return x, y
def disentangle(x, w, j):
    x = x.transpose(0,3,2,1) # [DGCN-TRL,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return xl, xh


def load_data(filepath, P, Q, train_ratio, test_ratio, log):
    Traffic = np.load(filepath)['data'][..., :1]
    num_step, num_nodes, _ = Traffic.shape
    time_idx = np.arange(num_step)
    tod = time_idx % 288
    dow = (time_idx//288) % 7
    TE = np.stack([dow,tod], axis=-1)

    TE_tile = np.tile(TE[:, np.newaxis, :], (1, num_nodes, 1))
    log_string(log, f'Shape of data: {Traffic.shape}')

    train_idx = int(train_ratio * num_step)
    val_idx = num_step - int(test_ratio * num_step)
    def process_split(data,te):
        x, y = seq2instance(data, P, Q)
        x_te, y_te = seq2instance(te, P, Q)
        return x, y, np.concatenate([x_te, y_te], axis=1)

    trainX, trainY, trainTE = process_split(Traffic[:train_idx], TE_tile[:train_idx])
    valX, valY, valTE = process_split(Traffic[train_idx:val_idx], TE_tile[train_idx:val_idx])
    testX, testY, testTE = process_split(Traffic[val_idx:], TE_tile[val_idx:])

    mean, std = np.mean(trainX), np.std(trainX)

    log_string(log, f'Shape of Train Data: {trainX.shape}')
    log_string(log, f'Shape of Validation Data: {valX.shape}')
    log_string(log, f'Shape of Test Data: {testX.shape}')

    return trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE, mean, std, Traffic[:train_idx, ..., 0]
