import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet
from dataset import get_train_data

from loss_scaler import DynamicLossScaler, StaticLossScaler


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    log_interval: int = 10,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        ################
        # loss scaling #
        ################
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()

        found_inf = scaler.unscale_(model.parameters())
        if not found_inf:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)

        scaler.update(found_inf)
        ################

        # outputs are logits, prob > 0.5 means logit > 0.
        with torch.no_grad():
            preds = outputs > 0.
            targets = labels > 0.5
            accuracy = (preds == targets).float().mean()

        if step % log_interval == 0:
            extra = f" scale={scaler.scale:.1f}"
            if found_inf:
                extra += " OVERFLOW"
            pbar.set_description(
                f"Loss: {loss.item():.4f}  Accuracy: {accuracy.item() * 100:.2f}%{extra}"
            )


def train(mode: str = "dynamic") -> None:
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if mode == "static":
        scaler = StaticLossScaler(scale=8192.)
    elif mode == "dynamic":
        scaler = DynamicLossScaler()
    else:
        raise ValueError("mode must be one of: 'static', 'dynamic'")

    train_loader = get_train_data()

    num_epochs = 5
    for _ in range(num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "dynamic"], default="dynamic")
    args = parser.parse_args()

    train(mode=args.mode)
