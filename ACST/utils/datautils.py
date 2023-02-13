import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_time_features(dt: pd.DataFrame.index):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float64)


def load_forecast_csv(config: dict):
    saved_data = config['saved_data']
    name = config['name']
    univar = config['univar']
    univar_feature = config['univar_feature']

    data = pd.read_csv(f'{saved_data}/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = get_time_features(data.index)
    n_covariant_cols = dt_embed.shape[-1]
    col_names = data.columns.values.tolist()

    if univar:
        assert univar_feature in col_names, f'Error! Found No feature in {col_names}'
        data = data[[univar_feature]]
        pass

    data = data.to_numpy()

    train_ratio = config['train_ratio']
    valid_ratio = config['valid_ratio']
    test_ratio = config['test_ratio']

    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    elif name.startswith('M5'):
        train_slice = slice(None, int(0.8 * (1913 + 28)))
        valid_slice = slice(int(0.8 * (1913 + 28)), 1913 + 28)
        test_slice = slice(1913 + 28 - 1, 1913 + 2 * 28)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    if name in ['electricity'] or name.startswith('M5'):
        data = np.expand_dims(data.T, -1)
        pass
    else:
        data = np.expand_dims(data.T, -1)
        pass

    if n_covariant_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
        pass

    return data, train_slice, valid_slice, test_slice, scaler, n_covariant_cols, col_names
