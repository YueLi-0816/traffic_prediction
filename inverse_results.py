import numpy as np
from data.data_loader import Dataset_Custom
from utils.metrics import metric
import os


def Inverse_Results(setting, args):
    """
    Convert saved scaled predictions (pred.npy, true.npy) back to original scale.
    """
    # Load stored pred & true
    result_path = f'./results/{setting}/'
    preds = np.load(result_path + 'pred.npy')
    trues = np.load(result_path + 'true.npy')

    # Rebuild the dataset to recover the SCALER
    # We rebuild Dataset_Custom ONLY to access its scaler;
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',                    # scaler is fit on TRAIN SPLIT
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=0,
        freq=args.freq,
        cols=args.cols
    )

    scaler = dataset.scaler  # this is the original scaler used during training

    # Apply inverse scaling: reshape → inverse → reshape back
    def _inverse(x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])    # 2D for StandardScaler
        x2 = scaler.inverse_transform(x2)
        return x2.reshape(orig_shape)

    preds_inv = _inverse(preds)
    trues_inv = _inverse(trues)

    # Calculate metrics on ORIGINAL scale
    mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)

    # Save inverse results
    np.save(result_path + 'pred_inverse.npy', preds_inv)
    np.save(result_path + 'true_inverse.npy', trues_inv)
    np.save(result_path + 'metrics_inverse.npy', np.array([mae, mse, rmse, mape, mspe]))

    print("✓ Inverse transformation complete")
    print("=== Inverse Scale Metrics ===")
    print(f"MSE:  {mse}")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
    print(f"MSPE: {mspe}")

    return preds_inv, trues_inv, (mae, mse, rmse, mape, mspe)