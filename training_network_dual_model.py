import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances
from triplet_model import GeM, interpolate_pos_embed
from dinov2.models.vision_transformer import vit_small as dinov2_vits
from dino.vision_transformer import vit_small as gastronet_vits
from peft.tuners.lora import LoraModel
from peft import LoraConfig, TaskType
import os


# ----------------- LoRA Setup -----------------
def add_lora_to_vit(vit, r=8, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["qkv", "proj", "fc1", "fc2"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )
    lora_wrapped = LoraModel(vit, cfg, adapter_name="default")
    return lora_wrapped


# ----------------- Pooling Layer -----------------
def get_pooling_layer(pooling_type="gem"):
    pooling_type = pooling_type.lower()
    if pooling_type == "avg":
        print("📘 Using Average Pooling")
        return nn.AdaptiveAvgPool2d((1, 1))
    elif pooling_type == "max":
        print("📘 Using Max Pooling")
        return nn.AdaptiveMaxPool2d((1, 1))
    else:
        print("📘 Using GeM Pooling (default)")
        return GeM(p=3.0, trainable=True)


# ----------------- Backbone Models -----------------
def build_gastronet_model(pooling="gem"):
    model = gastronet_vits(patch_size=16)
    model.global_pool = get_pooling_layer(pooling)
    model.head = nn.Identity()
    sd = torch.load("GastroNet_weights/VITS_GastroNet-5M_DINOv1.pth", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model = add_lora_to_vit(model, r=8, alpha=16, dropout=0.05)
    return model


def build_dinov2_model(pooling="gem"):
    model = dinov2_vits(patch_size=14)
    model.global_pool = get_pooling_layer(pooling)
    model.head = nn.Identity()
    sd = torch.load("pretrained_weights/dinov2_vits14_reg4_pretrain.pth", map_location="cpu")
    if "teacher" in sd:
        sd = sd["teacher"]
    interpolate_pos_embed(model, sd)
    model.load_state_dict(sd, strict=False)
    model = add_lora_to_vit(model, r=8, alpha=16, dropout=0.05)
    return model


# ----------------- Dual Backbone Wrapper -----------------
class DualBackboneWrapper(nn.Module):
    def __init__(self, model1, model2, out_dim=512):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.fusion_fc = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        f1 = F.normalize(self.model1(x), dim=1)  # (B, 384) normalized
        f2 = F.normalize(self.model2(x), dim=1)  # (B, 384) normalized
        z = self.fusion_fc(torch.cat([f1, f2], dim=1))  # (B, 512)
        return F.normalize(z, dim=1)  # (B, 512) normalized


# ----------------- Validation -----------------
@torch.no_grad()
def validate_triplet(model, val_loader, device, loss_fn, miner):
    model.eval()
    total_loss = 0.0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)

        indices = miner(embeddings, labels)

        # If no valid triplets found, compute loss without miner
        if isinstance(indices, tuple) and all(t.numel() == 0 for t in indices if t is not None):
            loss = loss_fn(embeddings, labels)
        else:
            loss = loss_fn(embeddings, labels, indices_tuple=indices)

        total_loss += loss.item()

    return total_loss / max(1, len(val_loader))


# ----------------- Training -----------------
def train_dual_model(
        train_loader,
        val_loader,
        device,
        epochs=30,
        margin=0.4,
        triplet_type="semihard",
        freeze_backbone=False,
        save_path="results/dual_branch/best_model.pth",
        backbone_lr=2e-4,
        head_lr=5e-4,
        weight_decay=1e-4,
        pooling="gem"
):
    """
    Train dual-branch model with Triplet Loss

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        margin: Margin for triplet loss
        triplet_type: Type of triplets ("semihard", "hard", "all")
        freeze_backbone: If True, only train LoRA parameters
        save_path: Path to save best model
        backbone_lr: Learning rate for backbone (LoRA)
        head_lr: Learning rate for fusion head
        weight_decay: Weight decay
        pooling: Pooling type ("gem", "avg", "max")
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Build models
    print("\n🔧 Building dual-branch model...")
    m1 = build_gastronet_model(pooling)
    m2 = build_dinov2_model(pooling)
    model = DualBackboneWrapper(m1, m2, out_dim=512).to(device)

    # Freeze backbone if specified
    if freeze_backbone:
        print("🔒 Freezing backbone (training only LoRA + fusion head)")
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Always train fusion head
        for p in model.fusion_fc.parameters():
            p.requires_grad = True

    # Optimizer
    param_groups = [
        {"params": filter(lambda p: p.requires_grad, m1.parameters()),
         "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": filter(lambda p: p.requires_grad, m2.parameters()),
         "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.fusion_fc.parameters(),
         "lr": head_lr, "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss and miner
    distance = distances.CosineSimilarity()
    loss_fn = losses.TripletMarginLoss(margin=0.6, distance=distance)
    miner = miners.TripletMarginMiner(margin=margin, distance=distance,
                                      type_of_triplets=triplet_type)

    best_val_loss = float("inf")
    best_model_state = None

    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Triplet type: {triplet_type}")
    print(f"   Margin: {margin}")
    print(f"   Pooling: {pooling}")
    print(f"   Freeze backbone: {freeze_backbone}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            embeddings = model(images)

            # Mine hard triplets
            indices = miner(embeddings, labels)
            num_triplets = indices[0].shape[0] if isinstance(indices, tuple) and len(indices) > 0 else 0

            # Compute loss
            if isinstance(indices, tuple) and all(t.numel() == 0 for t in indices if t is not None):
                loss = loss_fn(embeddings, labels)
            else:
                loss = loss_fn(embeddings, labels, indices_tuple=indices)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(triplets=num_triplets, loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, len(train_loader))
        scheduler.step()

        # Validation
        val_loss = validate_triplet(model, val_loader, device, loss_fn, miner)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"           ✓ New best model (val_loss: {val_loss:.4f})")

    # Save models
    final_path = save_path.replace(".pth", "_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Final model saved to: {final_path}")

    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"💾 Best model (val_loss: {best_val_loss:.4f}) saved to: {save_path}")

    print("\n✅ Training complete!")
    return model