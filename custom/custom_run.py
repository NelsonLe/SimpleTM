import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from model.CustomTM import CustomTM

# makes "samples" in a dataset where one sample input looks as far back as one length
# a label is one prediction legnth after
class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, length: int, prediction_length: int):
        self.data = data.astype(np.float32)
        self.length = length
        self.prediction_length = prediction_length

        self.num_windows = len(self.data) - self.length - self.prediction_length + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.length]
        y = self.data[idx + self.length : idx + self.length + self.prediction_length]
        return torch.tensor(x), torch.tensor(y)

def load_etth2_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])  # change type
    df = df.sort_values("date").set_index("date")  # sort date
    return df

def load_annual_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.pivot(index="date", columns="Country", values="Exchange rate")
    df = df.sort_index()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(limit_direction="both").ffill().bfill()  # fill missing
    return df

def preprocess_dataset(
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        length: int,
        prediction_length: int,
        batch_size: int,
        num_workers: int,
    ) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:

    # split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # scale
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    # convert to windowdataset
    loaders = {
        "train": DataLoader(
            WindowDataset(train_scaled, length, prediction_length),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        "val": DataLoader(
            WindowDataset(val_scaled, length, prediction_length),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ),
        "test": DataLoader(
            WindowDataset(test_scaled, length, prediction_length),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }

    meta = {"num_variables": df.shape[1], "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)}

    return loaders, meta

def metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mse": torch.mean((pred - target) ** 2), "mae": torch.mean(torch.abs(pred - target))}

# reproduceability
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# useful for oscar
def get_device(device_arg: str) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loaders(args):
    df = load_etth2_data(args.data_path) if args.dataset_type == "etth2" else load_annual_data(args.data_path)
    return preprocess_dataset(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        length=args.length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

# build customtm and feed it the variables
def build_model(args, num_variables: int, device: torch.device) -> CustomTM:
    model = CustomTM(args).to(device)
    return model

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, path)

def load_checkpoint(path: str, model: nn.Module):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer = None,
    ) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_mse, total_mae, total_count = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            batch_metrics = metrics(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_mse += batch_metrics["mse"].item() * batch_size
            total_mae += batch_metrics["mae"].item() * batch_size

    return {
        "loss": total_loss / total_count,
        "mse": total_mse / total_count,
        "mae": total_mae / total_count,
    }

def train(args):
    set_seed(args.seed)
    device = get_device(args.device)

    loaders, meta = make_loaders(args)
    model = build_model(args, meta["num_variables"], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    history = []

    print(f"Dataset: {args.dataset_type}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {device}")
    print(f"Variables: {meta['num_variables']}")
    print(f"Rows -> train={meta['train_rows']}, val={meta['val_rows']}, test={meta['test_rows']}")
    print(
        f"Config -> length={args.length}, pred={args.prediction_length}, pseudo={args.pseudo_length}, "
        f"batch={args.batch_size}, epochs={args.epochs}, lr={args.learning_rate}, m={args.m}"
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_stats = run_epoch(model, loaders["val"], criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_mse": train_stats["mse"],
                "train_mae": train_stats["mae"],
                "val_loss": val_stats["loss"],
                "val_mse": val_stats["mse"],
                "val_mae": val_stats["mae"],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_stats['loss']:.6f} mse={train_stats['mse']:.6f} mae={train_stats['mae']:.6f} | "
            f"val loss={val_stats['loss']:.6f} mse={val_stats['mse']:.6f} mae={val_stats['mae']:.6f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(best_model_path, model, optimizer, epoch)

    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"Saved training history to {history_path}")
    print(f"Saved best checkpoint to {best_model_path}")

    args.checkpoint_path = best_model_path
    test(args)

def test(args):
    set_seed(args.seed)
    device = get_device(args.device)

    loaders, meta = make_loaders(args)
    model = build_model(args, meta["num_variables"], device)
    criterion = nn.MSELoss()


    checkpoint = load_checkpoint(args.checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    test_stats = run_epoch(model, loaders["test"], criterion, device)

    print(
        f"Test | loss={test_stats['loss']:.6f} mse={test_stats['mse']:.6f} mae={test_stats['mae']:.6f}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, "test_metrics.csv")
    pd.DataFrame([{
        "test_loss": test_stats["loss"],
        "test_mse": test_stats["mse"],
        "test_mae": test_stats["mae"],
    }]).to_csv(metrics_path, index=False)
    print(f"Saved test metrics to {metrics_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CustomTM")

    # CustomTM model arguments
    # @ Nelson, can variables just be inferred from the dataset?
    # not sure how preprocessing works - TGR
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--dataset_type", choices=["etth2", "annual"], required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--variables", type=int, default=None, help="Number of variables in multivariate time series data")
    parser.add_argument("--length", type=int, default=None, help="Lookback window length")
    parser.add_argument("--pseudo_length", type=int, default=None, help="Dimensionality of pseudo-tokens")
    parser.add_argument("--prediction_length", type=int, default=None, help="Forecasting window length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio for non-attention feedforward layers")
    parser.add_argument("--m", type=int, default=None, help="Number of decomposition levels for Wavelet transform")
    parser.add_argument("--learnable_wavelets", action="store_true", dest="learnable_wavelets", help="Allow wavelet convolutions to be learnable")
    parser.add_argument("--wv", type=str, default="db1", help="Wavelet function used for decomposition/initialization")
    parser.add_argument("--pad_mode", type=str, default="circular", choices=["constant", "reflect", "replicate", "circular", "edge", "interpolate"], help="Padding mode for constant-length wavelet decomposition")
    parser.add_argument("--inverted", action="store_true", help="Apply linear projection after wavelet decomposition in original time domain")
    parser.add_argument("--alpha", type=float, default=1.0, help="Wedge product weighting in geometric self-attention")
    parser.add_argument("--scale", type=float, default=None, help="Scaling factor for query-key product in self-attention")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout ratio for attention layers")
    parser.add_argument("--normalize", action="store_true", dest="normalize", help="Feed normalized data into the model, then unnormalize outputs")
    parser.add_argument("--transformer_layers", type=int, default=1,help="Number of SWT/Attention/ISWT/Feedforward blocks")
    parser.add_argument("--is_geometric", action="store_true", dest="is_geometric",help="Use geometric (as opposed to vanilla) attention")
    parser.add_argument("--encoder_activation", type=str, default="gelu",choices=["relu", "gelu"],help="Activation function to use in feedforward layers")
    parser.add_argument("--feedforward_dim", type=int, default=32,help="Hidden dimension of feedforward layers")

    return parser
    #TODO: add the following flags...
    # l1 weight
    # early stopping rounds/epochs
    # among various others...
    # not sure exactly what is/isn't extraneous
    # - TGR

def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
