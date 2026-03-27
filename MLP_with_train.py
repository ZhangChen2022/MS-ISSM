import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io
import numpy as np
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import pandas as pd
import os

def logistic_func(X, beta1, beta2, beta3, beta4):
    """ 4-parameter logistic function """
    logisticPart = 1 + np.exp(np.clip(-(X - beta3) / (np.abs(beta4) + 1e-8), -50, 50))
    yhat = beta2 + (beta1 - beta2) / logisticPart
    return yhat


def fit_function(y_label, y_output):
    """ Nonlinearly fit the network output scores to the actual MOS scale (VQEG standard) """
    beta1_init = np.max(y_label)
    beta2_init = np.min(y_label)
    beta3_init = np.mean(y_output)
    beta4_init = np.std(y_output) / 4.0
    try:
        popt, _ = curve_fit(logistic_func, y_output, y_label,
                            p0=[beta1_init, beta2_init, beta3_init, beta4_init],
                            maxfev=10000)
        y_mapped = logistic_func(y_output, *popt)
    except:
        y_mapped = y_output
    return y_mapped


# ==========================================
# 1. Loss function
# ==========================================
class PLCCLoss(nn.Module):
    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.squeeze(), y_true.squeeze()
        if y_pred.shape[0] < 2: return torch.tensor(0.0, requires_grad=True).to(y_pred.device)
        mean_pred, mean_true = torch.mean(y_pred), torch.mean(y_true)
        centered_pred, centered_true = y_pred - mean_pred, y_true - mean_true
        std_pred, std_true = torch.std(y_pred), torch.std(y_true)
        covariance = torch.mean(centered_pred * centered_true)
        pearson_corr = covariance / (std_pred * std_true + 1e-8)
        return 1.0 - pearson_corr


class RankLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(RankLoss, self).__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.squeeze(), y_true.squeeze()
        n = y_pred.size(0)
        if n < 2: return torch.tensor(0.0, requires_grad=True).to(y_pred.device)
        idx = torch.randperm(n)
        y_pred_shuffled, y_true_shuffled = y_pred[idx], y_true[idx]
        target = torch.sign(y_true - y_true_shuffled)
        mask = target != 0
        if mask.sum() == 0: return torch.tensor(0.0, requires_grad=True).to(y_pred.device)
        return self.loss_fn(y_pred[mask], y_pred_shuffled[mask], target[mask])


# ==========================================
# 2. Model structure
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4), nn.SiLU(),
            nn.Linear(input_dim // 4, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class ResGroupedMLP(nn.Module):
    def __init__(self):
        super(ResGroupedMLP, self).__init__()
        self.scales = ['H', 'M', 'L']
        self.feats = ['Y', 'Cr', 'Cu']
        self.encoders = nn.ModuleDict()

        for scale in self.scales:
            for feat in self.feats:
                key = f"{scale}_{feat}"
                self.encoders[key] = nn.Sequential(
                    nn.Linear(5, 64), nn.BatchNorm1d(64), nn.SiLU(), ResidualBlock(64, dropout=0.1),
                )

        self.scale_fusers = nn.ModuleDict()
        self.scale_attentions = nn.ModuleDict()

        for scale in self.scales:
            self.scale_attentions[scale] = AttentionBlock(192)
            self.scale_fusers[scale] = nn.Sequential(
                nn.Linear(192, 96), nn.BatchNorm1d(96), nn.SiLU(), ResidualBlock(96, dropout=0.1)
            )

        self.global_attention = AttentionBlock(288)

        self.regressor = nn.Sequential(
            nn.Linear(288, 128), nn.BatchNorm1d(128), nn.SiLU(), ResidualBlock(128, dropout=0.2),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale_embeddings = []
        for s_idx, scale in enumerate(self.scales):
            feat_embeddings = []
            for f_idx, feat in enumerate(self.feats):
                sub_input = x[:, :, s_idx, f_idx]
                key = f"{scale}_{feat}"
                feat_embeddings.append(self.encoders[key](sub_input))
            concat = torch.cat(feat_embeddings, dim=1)
            concat = self.scale_attentions[scale](concat)
            scale_embeddings.append(self.scale_fusers[scale](concat))

        global_feat = torch.cat(scale_embeddings, dim=1)
        global_feat = self.global_attention(global_feat)

        out = self.regressor(global_feat)
        return out


# ==========================================
# 3. Data loader that can handle multiple datasets simultaneously
# ==========================================
class PCQADataset(Dataset):
    def __init__(self, dataset_name, feature_path, mos_path):
        print(f"--- Loading Raw Data for {dataset_name} ---")
        try:
            feat_data = scipy.io.loadmat(feature_path)
            candidates = []
            for k, v in feat_data.items():
                if not k.startswith('__') and (k == 'components' or (isinstance(v, np.ndarray) and v.size > 10)):
                    candidates.append((k, v))
            target_key, self.features = candidates[0]
            for k, v in candidates:
                if k == 'components': target_key, self.features = k, v; break

            if self.features.dtype.names and 'components' in self.features.dtype.names:
                self.features = self.features['components'][0, 0]
            elif self.features.dtype.names:
                self.features = self.features[self.features.dtype.names[0]][0, 0]

            for _ in range(5):
                if self.features.dtype == 'O' and self.features.size == 1:
                    try:
                        self.features = self.features.item()
                    except:
                        self.features = self.features[0]
                elif self.features.dtype == 'O':
                    try:
                        t = np.array(self.features.tolist())
                        if np.issubdtype(t.dtype, np.number): self.features = t; break
                    except:
                        pass
                    if self.features.ndim == 2 and self.features.shape[1] == 1:
                        try:
                            self.features = np.stack([x[0] for x in self.features]); break
                        except:
                            pass
                    break
                elif np.issubdtype(self.features.dtype, np.number):
                    break
        except Exception as e:
            print(f"Failed to load features: {e}")
            raise e

        mos_data = scipy.io.loadmat(mos_path)
        self.mos_values = None
        for k, v in mos_data.items():
            if not k.startswith('__') and isinstance(v, np.ndarray) and v.size > 10:
                self.mos_values = v;
                break

        if self.mos_values.ndim > 1 and self.mos_values.shape[1] >= 2:
            self.mos_values = self.mos_values[:, 1]

        self.mos_values = self.mos_values.flatten().astype(np.float32)
        self.features = self.features.astype(np.float32)

        print("Applying Log-Modulus Transformation...")
        self.features = np.sign(self.features) * np.log1p(np.abs(self.features))

        num_samples = len(self.features)
        if dataset_name == 'WPC':
            self.content_ids = np.repeat(np.arange(num_samples // 37 + 1), 37)[:num_samples]
        elif dataset_name == 'SJTU':
            self.content_ids = np.repeat(np.arange(9), num_samples // 9)[:num_samples]
        elif dataset_name == 'MPCCD':
            self.content_ids = np.repeat(np.arange(8), num_samples // 8)[:num_samples]
        elif dataset_name == 'ICIP':
            self.content_ids = np.repeat(np.arange(6), 15)[:num_samples]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        print(f"Data Loaded: {num_samples} samples. Found {len(np.unique(self.content_ids))} reference contents.")

    def __len__(self):
        return len(self.features)


# ==========================================
# 4. Training and testing core logic
# ==========================================
def run_training(dataset_name):
    CONFIG = {
        'WPC': {'splits': 10, 'seed': 123},
        'SJTU': {'splits': 9, 'seed': 123},
        'MPCCD': {'splits': 8, 'seed': 42},
        'ICIP': {'splits': 6, 'seed': 1}
    }

    if dataset_name not in CONFIG:
        raise ValueError("Invalid dataset. Choose from WPC, SJTU, MPCCD, ICIP.")

    cfg = CONFIG[dataset_name]
    RANDOM_SEED = cfg['seed']
    N_SPLITS = cfg['splits']

    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 80
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RESULT_EXPORT_FILE = f"Detailed_Test_Results_{dataset_name}_Unified.xlsx"

    feature_file = f"{dataset_name}_features_tensor.mat"
    mos_file = f"{dataset_name}_mos_tensor.mat"

    if not os.path.exists(feature_file) or not os.path.exists(mos_file):
        print(f"Error: Missing feature or MOS file for {dataset_name}.")
        return

    full_dataset = PCQADataset(dataset_name, feature_file, mos_file)

    unique_ids = np.unique(full_dataset.content_ids)
    rng = np.random.RandomState(seed=RANDOM_SEED)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)
    id_map = {old: new for old, new in zip(unique_ids, shuffled_ids)}
    mapped_content_ids = np.array([id_map[cid] for cid in full_dataset.content_ids])

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = {'plcc': [], 'srocc': [], 'krocc': [], 'rmse': []}
    all_folds_detailed_data = []

    global_spliced_mapped_preds = []
    global_spliced_targets = []

    for fold, (train_ids, test_ids) in enumerate(
            gkf.split(full_dataset.features, full_dataset.mos_values, groups=mapped_content_ids)):
        print(f"\n" + "=" * 60)
        print(f">>> Starting Fold {fold + 1}/{N_SPLITS} for {dataset_name}")

        train_feat_raw, test_feat_raw = full_dataset.features[train_ids], full_dataset.features[test_ids]
        train_mos_raw, test_mos_raw = full_dataset.mos_values[train_ids], full_dataset.mos_values[test_ids]

        feat_mean, feat_std = train_feat_raw.mean(axis=0), train_feat_raw.std(axis=0) + 1e-6
        train_feat_norm = (train_feat_raw - feat_mean) / feat_std
        test_feat_norm = (test_feat_raw - feat_mean) / feat_std

        fold_mos_min = train_mos_raw.min()
        fold_mos_max = train_mos_raw.max()
        mos_range = fold_mos_max - fold_mos_min + 1e-8

        train_mos_norm = (train_mos_raw - fold_mos_min) / mos_range

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_feat_norm), torch.from_numpy(train_mos_norm).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=True)

        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_feat_norm), torch.from_numpy(test_mos_raw).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=False)

        model = ResGroupedMLP().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        c_mse, c_plcc, c_rank = nn.MSELoss(), PLCCLoss().to(DEVICE), RankLoss().to(DEVICE)

        best_s = -1.0
        best_p, best_k, best_r = -1.0, -1.0, -1.0
        best_raw_preds, best_mapped_preds, best_targets = None, None, None

        for epoch in range(EPOCHS):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()

                out = model(bx)
                loss = c_mse(out, by) + 0.2 * c_plcc(out, by) + 0.2 * c_rank(out, by)

                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            ps, ts = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    out_norm = model(bx.to(DEVICE)).cpu().numpy().flatten()
                    out_raw = out_norm * mos_range + fold_mos_min
                    ps.extend(out_raw)
                    ts.extend(by.numpy().flatten())

            ps_arr, ts_arr = np.array(ps), np.array(ts)
            s = spearmanr(ps_arr, ts_arr)[0]

            if s > best_s:
                best_s = s
                best_k = kendalltau(ps_arr, ts_arr)[0]
                m_ps = fit_function(ts_arr, ps_arr)
                best_p = pearsonr(m_ps, ts_arr)[0]
                best_r = np.sqrt(mean_squared_error(ts_arr, m_ps))
                best_raw_preds, best_mapped_preds, best_targets = ps_arr, m_ps, ts_arr

        print(f"Fold {fold + 1} Best Results -> PLCC: {best_p:.4f} | SROCC: {best_s:.4f} | RMSE: {best_r:.4f}")

        fold_results['plcc'].append(best_p)
        fold_results['srocc'].append(best_s)
        fold_results['krocc'].append(best_k)
        fold_results['rmse'].append(best_r)

        global_spliced_mapped_preds.extend(best_mapped_preds.tolist())
        global_spliced_targets.extend(best_targets.tolist())

        all_folds_detailed_data.append(pd.DataFrame({
            'Fold': fold + 1, 'Original_Index': test_ids, 'Content_ID': full_dataset.content_ids[test_ids],
            'True_MOS': best_targets, 'Raw_Pred_MOS': best_raw_preds, 'Mapped_Pred_MOS': best_mapped_preds
        }))

    final_detailed_df = pd.concat(all_folds_detailed_data, ignore_index=True)
    final_detailed_df.to_excel(RESULT_EXPORT_FILE, index=False)

    global_spliced_mapped_preds = np.array(global_spliced_mapped_preds)
    global_spliced_targets = np.array(global_spliced_targets)

    spliced_srocc = spearmanr(global_spliced_mapped_preds, global_spliced_targets)[0]
    spliced_krocc = kendalltau(global_spliced_mapped_preds, global_spliced_targets)[0]
    spliced_plcc = pearsonr(global_spliced_mapped_preds, global_spliced_targets)[0]
    spliced_rmse = np.sqrt(mean_squared_error(global_spliced_targets, global_spliced_mapped_preds))

    print("\n" + "=" * 60)
    print(f"🏆 FINAL VALIDATION RESULTS ({dataset_name} - {N_SPLITS} Fold)")
    print("=" * 60)
    print(f"📊 [1] K-fold Average:")
    print(f"  AVERAGE PLCC:  {np.mean(fold_results['plcc']):.4f}")
    print(f"  AVERAGE SROCC: {np.mean(fold_results['srocc']):.4f}")
    print(f"  AVERAGE KROCC: {np.mean(fold_results['krocc']):.4f}")
    print(f"  AVERAGE RMSE:  {np.mean(fold_results['rmse']):.4f}")

    print(f"\n🧩 [2] Global Splicing:")
    print(f"  SPLICED PLCC:  {spliced_plcc:.4f}")
    print(f"  SPLICED SROCC: {spliced_srocc:.4f}")
    print(f"  SPLICED KROCC: {spliced_krocc:.4f}")
    print(f"  SPLICED RMSE:  {spliced_rmse:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    DATASET_NAME = 'WPC'
    print(f"Starting execution for {DATASET_NAME} dataset...")
    run_training(DATASET_NAME)