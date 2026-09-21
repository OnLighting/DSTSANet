import argparse
import os
import time
import numpy as np
import torch

torch.set_num_threads(12)
from libs.graph_utils import load_graph
from libs.utils import log_string, load_data, compute_loss, metric, disentangle, set_global_seed
from model.models import DSTSANet
import warnings

os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

warnings.filterwarnings(
    "ignore",
    message="Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED.*",
    category=UserWarning,
)


class Solver(object):
    def __init__(self, config, log):
        self.config = config
        self.log = log
        self.best_epoch = 0
        self.train_times = []
        log_string(self.log, '====================== Data ======================')

        (self.trainX, self.trainY, self.trainTE,
         self.valX, self.valY, self.valTE,
         self.testX, self.testY, self.testTE,
         self.mean, self.std, data) = load_data(
            self.config['traffic_file'],
            self.config['input_len'],
            self.config['output_len'],
            self.config['train_ratio'],
            self.config['test_ratio'],
            self.log)

        self.localadj, self.spawave, self.temwave = load_graph(
            self.config['adj_file'],
            self.config['tem_adj_file'],
            self.config['heads'] * self.config['dims'],
            data,
            self.log
        )

        self.best_epoch = 0
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_dataloaders()

    def build_model(self):
        self.model = DSTSANet(
            self.config['heads'], self.config['dims'], self.config['layers'],
            self.config['sample'], self.config['levels'], self.localadj,
            self.spawave, self.temwave, self.config['input_len'],
            self.config['output_len'],
            ablation=self.config.get('ablation', 'none')
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config['learning_rate']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=2e-6
        )

    def build_dataloaders(self):
        log_string(self.log, "wavelet transform")
        wave_type = self.config['wave']

        trainXL, trainXH = disentangle(self.trainX, wave_type, 1)
        trainYL, trainYH = disentangle(self.trainY, wave_type, 1)

        valXL, valXH = disentangle(self.valX, wave_type, 1)

        testXL, testXH = disentangle(self.testX, wave_type, 1)

        trainXL = (trainXL - self.mean) / self.std
        trainXH = (trainXH - self.mean) / self.std
        valXL = (valXL - self.mean) / self.std
        valXH = (valXH - self.mean) / self.std
        testXL = (testXL - self.mean) / self.std
        testXH = (testXH - self.mean) / self.std

        log_string(self.log, "build dataLoader")

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(trainXL).float(),
            torch.from_numpy(trainXH).float(),
            torch.from_numpy(trainYL).float(),
            torch.from_numpy(trainYH).float(),
            torch.from_numpy(self.trainY).float(),
            torch.from_numpy(self.trainTE[:, :, 0, :]).float()
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            shuffle=True, pin_memory=True, num_workers=0
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(valXL).float(),
            torch.from_numpy(valXH).float(),
            torch.from_numpy(self.valTE[:, :, 0, :]).float(),
            torch.from_numpy(self.valY).float()
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config['batch_size'],
            shuffle=False, pin_memory=True, num_workers=0
        )

        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(testXL).float(),
            torch.from_numpy(testXH).float(),
            torch.from_numpy(self.testTE[:, :, 0, :]).float(),
            torch.from_numpy(self.testY).float()
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config['batch_size'],
            shuffle=False, pin_memory=True, num_workers=0
        )

    def vali(self):
        self.model.eval()
        pred = []
        label = []

        with torch.no_grad():
            for XL, XH, TE, Y in self.val_loader:
                XL = XL.to(self.device, non_blocking=True)
                XH = XH.to(self.device, non_blocking=True)
                TE = TE.to(self.device, non_blocking=True)

                y_hat, _, _ = self.model(XL, XH, TE)

                pred.append(y_hat.detach().cpu().numpy() * self.std + self.mean)
                label.append(Y.numpy())

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        mae, rmse, mape = metric(pred, label)
        log_string(self.log, f"average, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}")

        return mae, rmse, mape

    def train(self):
        log_string(self.log, "====================== Train ======================")
        min_loss = 10000000.0
        early_stop_counter = 0
        patience = self.config['patience']

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, self.config['max_epoch'] + 1):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()

            for XL, XH, YL, YH, Y, TE in self.train_loader:
                XL = XL.to(self.device, non_blocking=True)
                XH = XH.to(self.device, non_blocking=True)
                YL = YL.to(self.device, non_blocking=True)
                YH = YH.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                TE = TE.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                y_hat, y_hat_l, y_hat_h = self.model(XL, XH, TE)

                loss, pred_loss, aux_loss, rex_loss = compute_loss(
                    Y=Y, YL=YL, YH=YH,
                    y_hat=y_hat, y_hat_l=y_hat_l, y_hat_h=y_hat_h,
                    std=self.std, mean=self.mean,
                    lambda_pred=self.config['lambda_pred'],
                    lambda_aux=self.config['lambda_aux'],
                    lambda_rec=self.config['lambda_rec']
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                train_l_sum += loss.detach().cpu().item()
                batch_count += 1

            epoch_time = time.time() - start
            ms_per_iter = (epoch_time / batch_count) * 1000
            self.train_times.append(ms_per_iter)

            log_string(self.log, f"epoch:{epoch},"
                                 f"lr:{self.optimizer.param_groups[0]['lr']:.6f},"
                                 f"loss:{(train_l_sum / batch_count):.4f},"
                                 f"time:{epoch_time:.1f}sec "
                                 f"({ms_per_iter:.1f}ms/iter)")

            mae, rmse, mape = self.vali()
            self.lr_scheduler.step(mae)

            if mae < min_loss:
                self.best_epoch = epoch
                min_loss = mae
                torch.save(self.model.state_dict(), self.config['model_file'])
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                log_string(self.log, f'Early stop counter: {early_stop_counter}/{patience}')
                if early_stop_counter >= patience:
                    log_string(self.log, f'Early stopping triggered! Model stopped improving for {patience} epochs.')
                    break

        log_string(self.log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(self.log, "====================== Test ======================")

        self.model.load_state_dict(torch.load(self.config['model_file'], map_location=self.device))
        self.model.eval()

        pred = []
        label = []

        with torch.no_grad():
            for XL, XH, TE, Y in self.test_loader:
                XL = XL.to(self.device, non_blocking=True)
                XH = XH.to(self.device, non_blocking=True)
                TE = TE.to(self.device, non_blocking=True)

                y_hat, _, _ = self.model(XL, XH, TE)

                pred.append(y_hat.detach().cpu().numpy() * self.std + self.mean)
                label.append(Y.numpy())

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)

        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, ...], label[:, i, ...])
            log_string(self.log, f"step:{i + 1}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}")

        mae_avg, rmse_avg, mape_avg = metric(pred, label)
        log_string(self.log, f"average, mae:{mae_avg:.4f}, rmse:{rmse_avg:.4f}, mape:{mape_avg:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="DF_STNet Training Script")

    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epoch', type=int, default=200, help='Max training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')

    parser.add_argument('--input_len', type=int, default=12, help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=12, help='Output sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (unused for training but needed for split)')
    parser.add_argument('--steps_per_day', type=int, default=288, help='Time steps per day')

    parser.add_argument('--dims', type=int, default=16, help='Model dimensions')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--sample', type=int, default=1, help='Sample factor')
    parser.add_argument('--wave', type=str, default='db2', help='Wavelet type')
    parser.add_argument('--levels', type=int, default=1, help='Wavelet levels')

    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'attn_shared', 'conv_shared'],
                        help='Ablation mode for encoder: '
                             'none (default DSTSANet), '
                             'attn_shared (both branches use Temporal Attention), '
                             'conv_shared (both branches use Temporal Conv)')

    parser.add_argument('--lambda_pred', type=float, default=1.0, help='Weight for prediction loss')
    parser.add_argument('--lambda_aux', type=float, default=0.4, help='Weight for auxiliary loss')
    parser.add_argument('--lambda_rec', type=float, default=0.05, help='Weight for reconstruction loss')

    parser.add_argument('--traffic_file', type=str, default='./data/PeMS07/PeMS07.npz', help='Traffic data file path')
    parser.add_argument('--adj_file', type=str, default='./data/PeMS07/adj.npy', help='Adjacency matrix file path')
    parser.add_argument('--tem_adj_file', type=str, default='./data/PeMS07/tem_adj.npy',
                        help='Temporal adjacency matrix file path')
    parser.add_argument('--model_file', type=str, default='./work_dirs/PeMS07/PeMSD7.pth', help='Path to save the model')
    parser.add_argument('--log_file', type=str, default='./log/PeMS07/log_train.txt', help='Log file path')
    return parser.parse_args()

def main():
    config = vars(parse_args())
    set_global_seed(config['seed'])
    os.makedirs(os.path.dirname(config['model_file']), exist_ok=True)
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    log_file = open(config['log_file'], 'w')
    log_string(log_file, "=================== Training Start ===================")
    log_string(log_file, str(config))
    solver = Solver(config, log_file)
    solver.train()
    solver.test()
    log_string(log_file, "=================== Training Finished ===================")
    log_file.close()
if __name__ == '__main__':
    main()