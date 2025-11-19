import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from triplet_model import GeM, interpolate_pos_embed
from dinov2.models.vision_transformer import vit_small as dinov2_vits
from dino.vision_transformer import vit_small as gastronet_vits
from torch import nn
import torch.nn.functional as F
import argparse
from peft.tuners.lora import LoraModel
from peft import LoraConfig, TaskType


# ==============================
# LoRA Setup
# ==============================
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


# ==============================
# Model Definition
# ==============================
class DualBackboneWrapper(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.fusion_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.model1(x)
        feat2 = self.model2(x)
        feat_cat = torch.cat([feat1, feat2], dim=1)
        fused_feat = self.fusion_fc(feat_cat)
        return F.normalize(fused_feat, dim=1)


# ==============================
# Utility Functions
# ==============================
def top_k_similar_with_scores(query_feats, gallery_feats, gallery_paths, k=31):
    sims = cosine_similarity(query_feats, gallery_feats)
    k = min(k, sims.shape[1])
    top_k_idx = np.argsort(-sims, axis=1)[:, :k]
    top_k_paths = [[gallery_paths[i] for i in row] for row in top_k_idx]
    top_k_scores = np.take_along_axis(sims, top_k_idx, axis=1)
    return top_k_paths, top_k_scores, sims


def save_image_safe(src_path: str, dst_path: str):
    img = Image.open(src_path)
    if dst_path.lower().endswith((".jpg", ".jpeg")):
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
    img.save(dst_path)


def save_results(query_paths, top_paths, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    for q_path, topk in zip(query_paths, top_paths):
        q_name = os.path.splitext(os.path.basename(q_path))[0]
        q_dir = os.path.join(out_dir, f"{q_name}_top{len(topk)}")
        os.makedirs(q_dir, exist_ok=True)
        save_image_safe(q_path, os.path.join(q_dir, "query.jpg"))
        for j, p in enumerate(topk):
            top_name = os.path.splitext(os.path.basename(p))[0]
            save_image_safe(p, os.path.join(q_dir, f"top{j + 1}_{top_name}.jpg"))


def build_dual_model(device, checkpoint, gastronet_path, dinov2_path):
    # Build GastroNet with LoRA
    m1 = gastronet_vits(patch_size=16)
    m1.global_pool = GeM(p=3.0, trainable=True)
    m1.head = nn.Identity()
    m1.load_state_dict(torch.load(gastronet_path, map_location="cpu"), strict=False)
    m1 = add_lora_to_vit(m1, r=8, alpha=16, dropout=0.05)  # ✅ LoRA 추가!

    # Build DINOv2 with LoRA
    m2 = dinov2_vits(patch_size=14)
    m2.global_pool = GeM(p=3.0, trainable=True)
    m2.head = nn.Identity()
    state = torch.load(dinov2_path, map_location="cpu")
    if "teacher" in state:
        state = state["teacher"]
    interpolate_pos_embed(m2, state)
    m2.load_state_dict(state, strict=False)
    m2 = add_lora_to_vit(m2, r=8, alpha=16, dropout=0.05)  # ✅ LoRA 추가!

    # Build DualBackbone and load trained weights
    model = DualBackboneWrapper(m1, m2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    return model.eval()


def get_image_paths(folder, sample_every=1):
    """Get all image paths from a folder with optional sampling"""
    exts = (".jpg", ".jpeg", ".png")
    fnames = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    if not fnames:
        return []
    if sample_every > 1:
        idx = range(0, len(fnames), sample_every)
        fnames = [fnames[i] for i in idx]
    return [os.path.join(folder, f) for f in fnames]


def get_all_image_paths_recursive(root_folder, sample_every=1):
    """Get all image paths from root folder and all subfolders"""
    all_paths = []

    # Check if there are subfolders
    subfolders = [d for d in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, d))]

    if subfolders:
        # Case 1: Has subfolders - collect from all subfolders
        print(f"Found {len(subfolders)} subfolders, collecting images from all...")
        for sub in sorted(subfolders):
            sub_path = os.path.join(root_folder, sub)
            sub_images = get_image_paths(sub_path, sample_every)
            all_paths.extend(sub_images)
            print(f"  - {sub}: {len(sub_images)} images")
    else:
        # Case 2: No subfolders - collect directly from root
        print(f"No subfolders found, collecting images directly from {root_folder}")
        all_paths = get_image_paths(root_folder, sample_every)

    return all_paths


def load_images_from_paths(paths, transform):
    if len(paths) == 0:
        raise ValueError("No image paths to load.")
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs), paths


def extract_embeddings(model, images, device, batch_size=32):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            feats.append(model(batch).cpu())
    return torch.cat(feats, dim=0)


# ==============================
# Argument Parser
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Branch Medical Image Retrieval")

    # Required arguments
    parser.add_argument("--query_dir", type=str, required=True,
                        help="Path to query images folder")
    parser.add_argument("--db_root", type=str, required=True,
                        help="Path to database root folder (with or without subfolders)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for retrieved images (default: outputs)")
    parser.add_argument("--gastronet_weights", type=str,
                        default="pretrained_weights/VITS_GastroNet-5M_DINOv1.pth",
                        help="Path to GastroNet pretrained weights")
    parser.add_argument("--dinov2_weights", type=str,
                        default="pretrained_weights/dinov2_vits14_reg4_pretrain.pth",
                        help="Path to DINOv2 pretrained weights")

    # Retrieval parameters
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Sample every N images from database (default: 1, no sampling)")
    parser.add_argument("--topk", type=int, default=31,
                        help="Number of top-k images to retrieve (default: 31)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for loading gallery images (default: 256)")
    parser.add_argument("--emb_batch_size", type=int, default=32,
                        help="Batch size for model inference (default: 32)")

    return parser.parse_args()


# ==============================
# Main Script
# ==============================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = build_dual_model(device, args.checkpoint, args.gastronet_weights, args.dinov2_weights)

    # Extract query embeddings
    print(f"\nLoading query images from {args.query_dir}")
    q_files = get_image_paths(args.query_dir)
    if len(q_files) == 0:
        raise ValueError(f"No images found in {args.query_dir}")
    print(f"Found {len(q_files)} query images")

    q_imgs, q_paths = load_images_from_paths(q_files, transform)
    q_feats = extract_embeddings(model, q_imgs, device, batch_size=args.emb_batch_size).numpy()

    # Collect all database images (from subfolders or directly)
    print(f"\nCollecting database images from {args.db_root}")
    g_paths = get_all_image_paths_recursive(args.db_root, sample_every=args.sample_every)

    if len(g_paths) == 0:
        raise ValueError(f"No images found in {args.db_root}")

    print(f"\nTotal database images: {len(g_paths)}")

    # Extract gallery embeddings in batches
    print("\nExtracting gallery embeddings...")
    g_feats_chunks = []
    for i in range(0, len(g_paths), args.batch_size):
        batch_paths = g_paths[i:i + args.batch_size]
        g_imgs_b, _ = load_images_from_paths(batch_paths, transform)
        g_feats_chunks.append(extract_embeddings(model, g_imgs_b, device,
                                                 batch_size=args.emb_batch_size).numpy())
        print(f"  Processed {min(i + args.batch_size, len(g_paths))}/{len(g_paths)} images")

    g_feats = np.concatenate(g_feats_chunks, axis=0)

    # Retrieve top-k similar images
    print(f"\nRetrieving top-{args.topk} similar images for each query...")
    top_paths, top_scores, _ = top_k_similar_with_scores(q_feats, g_feats, g_paths, k=args.topk)

    # Save results
    print(f"\nSaving results to {args.output_dir}")
    save_results(q_paths, [p[:args.topk] for p in top_paths], out_dir=args.output_dir)

    print(f"\n✅ Done! All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()