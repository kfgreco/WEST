import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from .data_loader import GNNDataLoader, PatientDataset
from .model_v2 import PatientTransformer
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from .loss import contrastive_loss
from .patient_dataset import PatientDataset
import pandas as pd
import time
from datetime import datetime, timedelta


class EMA:
    def __init__(self, model, decay=0.9):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * \
                    self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Patient Transformer Model')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='./Patients_1204',
                        help='Base path for input data files')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--summary_file_name', type=str, default='patient_summary.csv',
                        help='Name of the summary file')
    parser.add_argument('--remove_gold_negative', action='store_true',
                        help='Whether to remove gold negative samples')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--output_type', type=str, default='mean',
                        help='Output type for classification (mean, max, first)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--print_every', type=int, default=1,
                        help='Print metrics every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')

    # Device and seed
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Model saving
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save model checkpoints')

    parser.add_argument('--model_name', type=str, default='patient_transformer_single.pt',
                        help='Name of the model checkpoint file')

    # Distributed training arguments
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=-1,
                        help='Number of distributed processes')

    # Contrastive learning parameters
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter for contrastive loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss')

    # Add new sampling parameter
    parser.add_argument('--pos_ratio', type=float, default=None,
                        help='Target ratio of positive samples (0-1). None for no resampling')

    # EMA parameters
    parser.add_argument('--ema_decay', type=float, default=0.9,
                        help='Decay rate for EMA model')
    parser.add_argument('--use_ema', action='store_true',
                        help='Whether to use EMA model for validation')

    # Add new parameters
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length for patient records')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top k codes to consider for each patient')

    # Add new parameters
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Whether to use data augmentation during training')

    # In parse_args() function, add new parameter
    parser.add_argument('--scheduler_type', type=str, default='linear', choices=['cosine', 'linear'],
                        help='Type of learning rate scheduler (cosine or linear)')
    
    parser.add_argument('--label_column', type=str, default='FINALPAH',
                    help='Which column to use as the training label (e.g., FINALPAH, KOMAP_calibrated)')

    args = parser.parse_args()
    return args


def train_model(args):
    """Main training function"""
    # Create save directory if it doesn't exist
    if args.local_rank in [-1, 0]:  # Only create directory on main process
        os.makedirs(args.save_dir, exist_ok=True)

    # Initialize distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()

    # Set random seed
    torch.manual_seed(args.seed + args.local_rank)

    # Load and process code embeddings and mapping
    # -----------------------------------------------------------------------------
    # Code Embedding and Mapping Files
    # -----------------------------------------------------------------------------
    # These two files work together to link domain-specific concept identifiers
    # (e.g., CCS procedure codes) to their corresponding embedding vectors.
    #
    # 1. Mapping.csv
    #    - Columns:
    #        CODE:  The unique concept identifier (e.g., "CCS:1", "CCS:2", etc.)
    #        Index: The zero-based index that corresponds to the same concept’s
    #                row position in the embedding file.
    #    - Purpose:
    #        Provides a lookup table to translate between concept codes and
    #        their vector positions in the embedding matrix.
    #
    # 2. Embeddings.csv
    #    - Rows:
    #        Each row represents a single concept (aligned with Mapping.csv order).
    #    - Columns:
    #        V1, V2, V3, ... : Continuous numerical features forming an
    #        n-dimensional embedding vector.
    #    - Purpose:
    #        Encodes semantic relationships between concepts — concepts with
    #        similar meanings or contexts will have similar vector representations.

    datax = pd.read_csv('.../Transformer/Input/Embeddings.csv')
    mapping = pd.read_csv('.../Transformer/Input/Mapping.csv')

    code_embeddings = torch.tensor(
        datax.iloc[:, 0:(datax.shape[1])].to_numpy(),
        dtype=torch.float32
    )

    # Initialize data loader with processed embeddings and mapping
    train_dataset = PatientDataset(
        data_dir=os.path.join(args.data_path),
        summary_file_name=args.summary_file_name,
        code_embeddings=code_embeddings,
        code_mapping=mapping,
        max_seq_len=args.max_seq_len,
        top_k=args.max_seq_len,
        training=True,
        pos_ratio=args.pos_ratio,
        use_augmentation=args.use_augmentation,
        remove_gold_negative=args.remove_gold_negative,
        label_column=args.label_column
    )

    val_dataset = PatientDataset(
        data_dir=os.path.join(args.data_path),
        summary_file_name=args.summary_file_name,
        code_embeddings=code_embeddings,
        code_mapping=mapping,
        max_seq_len=args.max_seq_len,
        top_k=args.max_seq_len,
        training=False,
        label_column=args.label_column
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset) if args.local_rank != -1 else None
    val_sampler = DistributedSampler(
        val_dataset, shuffle=False) if args.local_rank != -1 else None

    num_workers = int(os.environ.get("NUM_WORKERS", 0))  # default safe

    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Initialize model and move to device
    device = torch.device(args.local_rank if args.local_rank != -1 else 'cuda')
    model = PatientTransformer(
        d_model=args.hidden_dim,
        nhead=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        output_type=args.output_type
    ).to(device)

    # Wrap model with DDP
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])

    # Initialize EMA if enabled
    ema = None
    if args.use_ema and args.local_rank in [-1, 0]:
        ema = EMA(model.module if args.local_rank != -
                  1 else model, decay=args.ema_decay)
        ema.register()

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler with warmup
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=len(train_loader) * args.num_epochs,
            num_cycles=0.5
        )
    else:  # linear
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=len(train_loader) * args.num_epochs
        )

    # Initialize timing statistics
    epoch_times = []
    data_times = []
    model_times = []
    if args.local_rank != -1:
        torch.cuda.synchronize()  # 确保GPU同步
    start_time = time.time()

    # Add CUDA events after device initialization
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # Training loop
    best_val_auc = 0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        if args.local_rank != -1:
            torch.cuda.synchronize()
        epoch_start = time.time()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        train_predictions = []
        train_labels = []

        if args.local_rank in [-1, 0]:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print("Training:")

        data_time = 0
        model_time = 0
        data_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.local_rank != -1:
                torch.cuda.synchronize()
            data_time += time.time() - data_start

            # Move batch to device
            code_embeddings = batch['code_embeddings'].to(device)
            counts = batch['counts'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            labels = labels.float()

            if args.local_rank != -1:
                torch.cuda.synchronize()

            if torch.cuda.is_available():
                start_event.record()

            # Forward pass and loss calculation
            if args.contrastive_weight > 0:
                outputs, features = model(
                    code_embeddings, counts, attention_mask)
            else:
                outputs = model(code_embeddings, counts,
                                attention_mask, return_features=False)

            # Get features from EMA model (in eval mode) for positive pairs
            if args.use_ema and args.local_rank in [-1, 0] and args.contrastive_weight > 0:
                ema.apply_shadow()  # Apply EMA weights
                model.eval()
                with torch.no_grad():
                    _, ema_features = model(
                        code_embeddings, counts, attention_mask)
                model.train()
                ema.restore()  # Restore original weights
                cont_loss = contrastive_loss(
                    features, ema_features, args.temperature)
            else:
                cont_loss = torch.tensor(0.0).to(device)

            # Calculate BCE loss with soft labels
            bce_loss = criterion(outputs, labels)

            # Combine losses
            loss = bce_loss + args.contrastive_weight * cont_loss

            # Backward pass and optimization steps
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Update EMA model
            if args.use_ema and args.local_rank in [-1, 0]:
                ema.update()

            # Store predictions and update metrics
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

            # Print batch progress (only on main process)
            if args.local_rank in [-1, 0] and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = scheduler.get_last_lr()[0]

                # Calculate ETA
                elapsed_time = time.time() - start_time
                progress = (epoch * len(train_loader) + batch_idx +
                            1) / (args.num_epochs * len(train_loader))
                if progress > 0:
                    eta_seconds = elapsed_time * (1 - progress) / progress
                    eta = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta = "N/A"

                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    # Convert ms to seconds
                    model_time += start_event.elapsed_time(end_event) / 1000

                print(f"Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"BCE Loss: {bce_loss.item():.4f} | "
                      f"Contrastive Loss: {cont_loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Data time: {data_time/(batch_idx+1):.3f}s/batch | "
                      f"Model time: {model_time/(batch_idx+1):.3f}s/batch | "
                      f"ETA: {eta}")

            data_start = time.time()

        # Record timing statistics
        if args.local_rank != -1:
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        data_times.append(data_time)
        model_times.append(model_time)

        if args.local_rank in [-1, 0]:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f"\nEpoch timing statistics:")
            print(f"Total epoch time: {epoch_time:.2f}s")
            print(f"Average epoch time: {avg_epoch_time:.2f}s")
            print(f"Data loading time: {data_time:.2f}s")
            print(f"Model training time: {model_time:.2f}s")

        # Validation
        model.eval()
        if args.use_ema and args.local_rank in [-1, 0]:
            ema.apply_shadow()

        val_predictions = []
        val_labels = []
        val_loss = 0

        if args.local_rank in [-1, 0]:
            print("\nValidation:")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                code_embeddings = batch['code_embeddings'].to(device)
                counts = batch['counts'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                labels = labels.float()

                outputs, _ = model(code_embeddings, counts, attention_mask)
                val_loss += criterion(outputs, labels).item()
                
                # Append model predictions
                val_predictions.extend(outputs.cpu().numpy())
                
                # val_labels.extend(labels.cpu().numpy())
                
                # Use true binary labels for AUC
                val_labels.extend(batch['gold_label'].cpu().numpy())

                # Print validation progress
                if args.local_rank in [-1, 0] and (batch_idx + 1) % 10 == 0:
                    avg_val_loss = val_loss / (batch_idx + 1)
                    print(f"Batch {batch_idx + 1}/{len(val_loader)} | "
                          f"Loss: {avg_val_loss:.4f}")

        # Restore original model weights
        if args.use_ema and args.local_rank in [-1, 0]:
            ema.restore()

        # Gather predictions from all processes for metrics
        if args.local_rank != -1:
            train_predictions = gather_predictions(train_predictions)
            train_labels = gather_predictions(train_labels)
            val_predictions = gather_predictions(val_predictions)
            val_labels = gather_predictions(val_labels)




        # ------------------------------
        # AUC per fold using kfold_2
        # ------------------------------
        summary_df = pd.read_csv(os.path.join(args.data_path, args.summary_file_name))
        summary_df.set_index('ID', inplace=True)
        summary_df.index = summary_df.index.astype(str)
        
        # Get patient IDs from val_dataset
        val_patient_ids = val_dataset.patient_ids[:len(val_predictions)]  # ensure same order/length
        val_patient_ids = [str(pid) for pid in val_patient_ids]
        
        # Get kfold_2 assignments for each val patient
        kfold_vals = [summary_df.loc[pid]['kfold_2'] if pid in summary_df.index else None for pid in val_patient_ids]
        
        # Convert to numpy arrays
        import numpy as np
        val_predictions = np.array(val_predictions).flatten()
        val_labels = np.array(val_labels).flatten()
        kfold_vals = np.array(kfold_vals)
        
        # Create masks
        mask1 = kfold_vals == 1
        mask2 = kfold_vals == 2
        
        # Compute AUCs
        val_auc_all = roc_auc_score(val_labels, val_predictions)
        val_auc_fold1 = roc_auc_score(val_labels[mask1], val_predictions[mask1]) if mask1.sum() > 0 else float('nan')
        val_auc_fold2 = roc_auc_score(val_labels[mask2], val_predictions[mask2]) if mask2.sum() > 0 else float('nan')
        
        






        # Calculate training AUC
        # train_auc = roc_auc_score(train_labels, train_predictions)

        # Calculate validation AUC
        # print(val_labels[:10], [i for i in val_predictions[:10]])
        # val_auc = roc_auc_score(val_labels, val_predictions)

        # Only print metrics on main process
        if args.local_rank in [-1, 0]:
            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}")
                print(f"Training Loss: {total_loss/len(train_loader):.4f}")
                print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
                # print(f"Training AUC: {train_auc:.4f}")
                # print(f"Validation AUC: {val_auc:.4f}")
                print(f"\nValidation AUC (All):    {val_auc_all:.4f}")
                print(f"Validation AUC (Fold 1): {val_auc_fold1:.4f}")
                print(f"Validation AUC (Fold 2): {val_auc_fold2:.4f}")
                
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                

            # Save best model
            if val_auc_all > best_val_auc: # can replace val_auc_all with val_auc_fold1 or val_auc_fold2
                best_val_auc = val_auc_all # can replace val_auc_all with val_auc_fold1 or val_auc_fold2
                if args.local_rank != -1:
                    model_to_save = model.module
                else:
                    model_to_save = model
                print(f"New best validation AUC: {best_val_auc:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state': ema.shadow if args.use_ema and args.local_rank in [-1, 0] else None,
                    'best_val_auc': best_val_auc,
                }, f"{args.save_dir}/best_{args.model_name}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    print(f"Best validation AUC: {best_val_auc:.4f}")

def gather_predictions(predictions):
    """Gather predictions from all distributed processes."""
    if not dist.is_initialized():
        return predictions

    predictions = torch.tensor(predictions).cuda()
    gathered = [torch.zeros_like(predictions)
                for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, predictions)
    return torch.cat(gathered).cpu().numpy()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
