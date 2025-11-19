import os
import torch
from dataloader import get_loader_from_folder
from training_network_dual_model import train_dual_model
import argparse
import random, numpy as np
from sklearn.metrics import average_precision_score


@torch.no_grad()
def compute_retrieval_metrics(model, val_loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, y in val_loader:
        imgs = imgs.to(device)
        z = model(imgs)
        feats.append(z.cpu())
        labels.append(y)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)

    sims = feats @ feats.t()
    sims.fill_diagonal_(-1)
    preds = sims.argsort(dim=1, descending=True)

    recall1, recall5 = 0, 0
    aps = []

    for i in range(len(labels)):
        gt = labels[i].item()
        retrieved = labels[preds[i]]
        recall1 += (gt == retrieved[:1]).any().item()
        recall5 += (gt == retrieved[:5]).any().item()
        y_true = (retrieved == gt).numpy().astype(int)
        y_score = sims[i, preds[i]].numpy()
        if y_true.sum() > 0:
            aps.append(average_precision_score(y_true, y_score))
    recall1 /= len(labels)
    recall5 /= len(labels)
    mAP = np.mean(aps) if aps else 0
    return recall1, recall5, mAP


def seed_everything(sd=42):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Backbone Retrieval Training")

    parser.add_argument("--mode", choices=["tri"], required=True,
                        help="Training mode: 'tri' (Triplet)")

    parser.add_argument("--train_dir", type=str, help="path your train directory")
    parser.add_argument("--val_dir", type=str, help="path your validation directory")
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--triplet_type", type=str, default="semihard",
                        choices=["semihard", "hard", "all"])
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--miner_margin", type=float, default=0.2)

    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone (train only LoRA/GeM..)")
    parser.add_argument("--pooling", type=str, default="gem",
                        choices=["gem", "avg", "max"],
                        help="Pooling type for feature aggregation")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(43)

    # Load data
    train_loader = get_loader_from_folder(
        args.train_dir, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_loader_from_folder(
        args.val_dir, batch_size=args.batch_size, shuffle=False
    )

    # Train model
    trained_model = train_dual_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        margin=args.triplet_margin,
        triplet_type=args.triplet_type,
        mode=args.mode,
        freeze_backbone=args.freeze_backbone,
        save_path=args.save_path,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        pooling=args.pooling,
    )

    # Save final model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), args.save_path)
    print(f"\n✅ Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
