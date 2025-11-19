import argparse, os
import torch
import torch.nn.functional as F

from evaluation import evaluate_retrieval
from dataloader import get_test_loader
from training_network_dual_model import DualBackboneWrapper, build_gastronet_model, build_dinov2_model

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate dual-backbone retrieval")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--query_dir", type=str, required=True, help="Query images root")
    p.add_argument("--db_dir", type=str, required=True, help="Database images root")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pooling", type=str, default="gem", choices=["gem", "avg", "max"],
                   help="Pooling type used during training (must match ckpt)")
    p.add_argument("--strict_load", action="store_true",
                   help="If set, strict=True when loading state_dict. Default is False for robustness.")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build models with the SAME pooling as training
    print(f"📘 Using pooling: {args.pooling}")
    m1 = build_gastronet_model(args.pooling)
    m2 = build_dinov2_model(args.pooling)
    model = DualBackboneWrapper(m1, m2).to(device)
    model.eval()

    # 3) Load checkpoint
    print(f"🔑 Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=args.strict_load)

    # 4) Dataloaders
    q_loader, g_loader = get_test_loader(
        query_dir=args.query_dir,
        gallery_dir=args.gallery_dir,
        batch_size=args.batch_size
    )
    idx_to_class = {v: k for k, v in q_loader.dataset.class_to_idx.items()}

    # 5) Evaluate
    evaluate_retrieval(model, q_loader, g_loader, device, idx_to_class=idx_to_class)

if __name__ == "__main__":
    main()
