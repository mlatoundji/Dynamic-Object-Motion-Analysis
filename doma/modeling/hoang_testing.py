import torch

from doma.dataset import GestureDataset, create_dataloaders, collate_gesture_batch
from doma.dataset import create_tsgcn_dataloaders

# train_loader, val_loader, test_loader = create_dataloaders(
#     manifest_path="data/processed/manifest.csv",
#     root_dir=".",
#     batch_size=32,
#     num_workers=4,
#     max_len=120
# )

# batch = next(iter(train_loader))

# print(batch["pose"].shape)
# print(batch["optflow"].shape)
# print(batch["label"].shape)
# # print(batch)
# print(batch["label"])

# # print batch keys
# print(batch.keys())

def check_batch(batch, epoch, batch_idx):
    """Kiểm tra batch có vấn đề gì không"""
    
    skeleton = batch["skeleton"]
    label = batch["label"]
    
    print(f"\n=== DEBUG BATCH {batch_idx} ===")
    print(f"skeleton stats: min={skeleton.min():.4f}, max={skeleton.max():.4f}, mean={skeleton.mean():.4f}")
    print(f"skeleton contains NaN: {torch.isnan(skeleton).any()}")
    print(f"skeleton contains Inf: {torch.isinf(skeleton).any()}")
    print(f"label stats: min={label.min()}, max={label.max()}")
    print(f"unique labels: {torch.unique(label)}")
    
    # Kiểm tra xem có label nào ngoài range không
    if label.max() >= 21 or label.min() < 0:
        print(f"⚠️ WARNING: Label out of range! num_classes=21")
    
    return torch.isnan(skeleton).any() or torch.isinf(skeleton).any()
    
def test_tsgcn_dataloader():
    """
    Test function to verify TSGCN dataloader output format.
    """
    # Create dataloaders (adjust paths as needed)
    train_loader, val_loader, test_loader = create_tsgcn_dataloaders(
        manifest_path="data/processed/manifest.csv",
        root_dir=".",
        batch_size=32,
        num_workers=0,  # Use 0 for debugging
        max_len=None,   # Don't truncate
        split_mode="train_val_test",
    )
    
    # Test train loader
    if train_loader:
        print("Testing train loader:")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  skeleton: {batch['skeleton'].shape}  # (B, 3, T, 21)")
            print(f"  motion: {batch['motion'].shape}      # (B, 3, T, 21)")
            print(f"  track: {batch['track'].shape}        # (B, T, 9)")
            print(f"  optflow: {batch['optflow'].shape}    # (B, T, 6)")
            print(f"  label: {batch['label'].shape}        # (B,)")
            print(f"  length: {batch['length']}")          # Original lengths
            print(f"  valid_mask: {batch['valid_mask'].shape}  # (B, T)")
            print(f"  sample_ids: {len(batch['sample_id'])} samples")
            
            # Verify skeleton values are in valid range
            valid_frames = batch['valid_mask'][0].sum().item()
            print(f"  First sample: {valid_frames} valid frames")
            
            if batch_idx == 0:  # Only test first batch
                break
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = test_tsgcn_dataloader()

def debug_dataloader(loader, num_batches=5, num_classes=None):
    """Debug dataloader bằng cách inspect vài batches đầu"""
    
    print("="*60)
    print("DEBUG DATALOADER")
    print("="*60)
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        print(f"\n--- Batch {batch_idx} ---")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  shape: {value.shape}")
                print(f"  dtype: {value.dtype}")
                print(f"  stats: min={value.min():.4f}, max={value.max():.4f}")
                
                if value.dtype in [torch.float16, torch.float32, torch.float64]:
                    print(f"  mean={value.mean():.4f}")
                
                print(f"  contains NaN: {torch.isnan(value).any()}")
                print(f"  contains Inf: {torch.isinf(value).any()}")
                
                if key == "label" and num_classes:
                    unique_labels = torch.unique(value)
                    print(f"  unique labels: {unique_labels.tolist()}")
                    if unique_labels.max() >= num_classes:
                        print(f"  ⚠️ Labels exceed num_classes ({num_classes})!")
                    if unique_labels.min() < 0:
                        print(f"  ⚠️ Labels contain negative values!")
        
        print("-"*40)
# debug_dataloader(train_loader, num_batches=3, num_classes=21)