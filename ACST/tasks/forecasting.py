import torch
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, ElasticNet, ElasticNetCV, MultiTaskElasticNetCV
from tqdm import trange
import time

from utils.otherutils import torch_pad_nan


def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([data[:, i:1 + n + i - pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
           labels.reshape(-1, labels.shape[2] * labels.shape[3])


def predict(config: dict):
    model = config['model']
    data = config['data']
    train_slice = config['train_slice']
    valid_slice = config['valid_slice']
    test_slice = config['test_slice']
    scaler = config['scaler']
    pred_lens = config['pred_lens']
    n_covariant_cols = config['n_covariant_cols']
    padding = config['padding']
    mode = config['mode']
    device = config['device']

    all_repr = encode(
        model=model,
        data=data,
        device=device,
        mode=mode,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )

    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariant_cols:]
    valid_data = data[:, valid_slice, n_covariant_cols:]
    test_data = data[:, test_slice, n_covariant_cols:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}

    for pred_len in pred_lens:
        print(f'horizon = {pred_len} steps')
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr, best_alpha = fit_ridge(train_features, train_labels, valid_features, valid_labels)  # 这里的lr是线性回归的意思
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        if test_data.shape[0] != 1:
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)
            pass
        else:
            test_pred = test_pred.reshape((-1, test_data.shape[2]))
            test_labels = test_labels.reshape((-1, test_data.shape[2]))

        if test_data.shape[0] > 1:
            test_pred = test_pred.swapaxes(0, 3).squeeze()
            test_labels = test_labels.swapaxes(0, 3).squeeze()
            test_pred = test_pred.transpose(1, 0, 2)
            test_labels = test_labels.transpose(1, 0, 2)

            test_pred_inv = torch.zeros_like(torch.from_numpy(test_pred)).numpy()
            test_labels_inv = torch.zeros_like(torch.from_numpy(test_labels)).numpy()
            n = test_labels.shape[0]
            for i in range(n):
                test_pred_inv[i, :, :] = scaler.inverse_transform(test_pred[i, :, :])
                test_labels_inv[i, :, :] = scaler.inverse_transform(test_labels[i, :, :])
                pass

        ours_result[pred_len] = {
            'raw': cal_metrics(test_pred, test_labels)
        }

        pass

    return ours_result


def encode(model, data, device, mode, mask=None, encoding_window=None, casual=False, sliding_length=None,
           sliding_padding=0,
           batch_size=128):
    print("encoding...")
    if mode == 'forecasting':
        encoding_window = None
        slicing = None
    else:
        raise NotImplementedError(f"mode {mode} has not been implemented")

    assert data.ndim == 3
    n_samples, ts_l, _ = data.shape
    dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        output = []
        for batch in loader:
            x = batch[0]
            if sliding_length is not None:
                reprs = []
                if n_samples < batch_size:
                    calc_buffer = []
                    calc_buffer_l = 0
                for i in trange(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not casual else 0)
                    x_sliding = torch_pad_nan(
                        x[:, max(l, 0): min(r, ts_l)],
                        left=-l if l < 0 else 0,
                        right=r - ts_l if r > ts_l else 0,
                        dim=1
                    )
                    if n_samples < batch_size:
                        if calc_buffer_l + n_samples > batch_size:
                            out = _eval_with_pooling(
                                model.encoder_q,
                                torch.cat(calc_buffer, dim=0),
                                device,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                        calc_buffer.append(x_sliding)
                        calc_buffer_l += n_samples
                    else:
                        out = _eval_with_pooling(
                            model.encoder_q,
                            x_sliding,
                            device,
                            mask,
                            slicing=slicing,
                            encoding_window=encoding_window
                        )
                        reprs.append(out)

                if n_samples < batch_size:
                    if calc_buffer_l > 0:
                        out = _eval_with_pooling(
                            model.encoder_q,
                            torch.cat(calc_buffer, dim=0),
                            device,
                            mask,
                            slicing=slicing,
                            encoding_window=encoding_window
                        )
                        reprs += torch.split(out, n_samples)
                        calc_buffer = []
                        calc_buffer_l = 0

                out = torch.cat(reprs, dim=1)
                if encoding_window == 'full_series':
                    out = F.max_pool1d(
                        out.transpose(1, 2).contiguous(),
                        kernel_size=out.size(1),
                    ).squeeze(1)
            else:
                out = _eval_with_pooling(model.encoder_q, x, device, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)

            output.append(out)

        output = torch.cat(output, dim=0)
    return output.numpy()


def _eval_with_pooling(model, x, device, mask=None, slicing=None, encoding_window=None):
    out_t, out_s = model(x.to(device, non_blocking=True))  # l b t d
    out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)
    return rearrange(out.cpu(), 'b d -> b () d')


def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    others = [30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 1250, 1500, 1750, 2000]
    alphas = np.arange(0, 20, 0.5).tolist()
    for item in others:
        alphas.append(item)
    valid_results = []
    for i in trange(len(alphas)):
        alpha = alphas[i]
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr, best_alpha


def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
