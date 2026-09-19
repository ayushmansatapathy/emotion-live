import os
import sys
import time
import argparse
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.dataset import get_dataloaders, EMOTION_CLASSES
from src.model import build_model
from src.evaluate import evaluate_model, print_evaluation_summary

def train_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            batch_acc = correct / total
            batch_loss = total_loss / total
            print(f"  Batch {i+1:3d}/{len(dataloader):3d} - Loss: {batch_loss:.4f} Acc: {batch_acc*100:.2f}%", flush=True)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training(config: dict):
    print("\n" + "="*70)
    print("STARTING TRAINING ITERATION WITH CONFIG:")
    print(json.dumps(config, indent=2))
    print("="*70)

    # Set threads
    if 'threads' in config:
        torch.set_num_threads(config['threads'])

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Dataloaders
    img_size = config.get('img_size', 112)
    batch_size = config.get('batch_size', 64)
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0
    )

    device = torch.device('cpu')

    # Loss function with class weights and label smoothing
    label_smoothing = config.get('label_smoothing', 0.1)
    use_weights = config.get('use_class_weights', True)
    criterion_weights = class_weights.to(device) if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weights, label_smoothing=label_smoothing)

    # Model
    backbone = config.get('backbone', 'mobilenet_v3_small')
    dropout = config.get('dropout', 0.3)
    model = build_model(backbone=backbone, num_classes=7, pretrained=True, dropout=dropout)
    
    resume_from = config.get('resume_from', None)
    if resume_from and os.path.exists(resume_from):
        print(f"Loading weights from prior checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=True)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
    
    model.to(device)

    # Phase 1: Warm-up head if specified
    warmup_epochs = config.get('warmup_epochs', 1)
    lr_head = config.get('lr_head', 1e-3)
    lr_backbone = config.get('lr_backbone', 3e-4)

    if warmup_epochs > 0:
        print(f"\n--- Phase 1: Head Warm-up ({warmup_epochs} epochs, lr={lr_head}) ---")
        model.freeze_backbone()
        optimizer_warmup = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-2)
        for ep in range(1, warmup_epochs + 1):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, criterion, optimizer_warmup, device=device)
            val_metrics = evaluate_model(model, val_loader, device=device, save_cm_path=None)
            dt = time.time() - t0
            print(f"Warm-up Epoch {ep}/{warmup_epochs} [{dt:.1f}s] - Train Loss: {loss:.4f} Acc: {acc*100:.2f}% | Val Acc: {val_metrics['accuracy']*100:.2f}% Macro-F1: {val_metrics['macro_f1']:.4f}")

    # Phase 2: Full or partial fine-tuning
    unfreeze_all = config.get('unfreeze_all', True)
    model.unfreeze_backbone(unfreeze_all=unfreeze_all)

    # Differential learning rates: smaller for backbone, larger for classifier head
    head_params = list(model.classifier.parameters())
    backbone_params = list(model.features.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': 1e-3},
        {'params': head_params, 'lr': lr_head, 'weight_decay': 1e-2}
    ])

    epochs = config.get('epochs', 5)
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_path = config.get('save_path', 'models/best_emotion_model.pth')

    print(f"\n--- Phase 2: Fine-tuning ({epochs} epochs) ---")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device=device)
        val_metrics = evaluate_model(model, val_loader, device=device, save_cm_path=None)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']
        dt = time.time() - t0

        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'config': config,
                'epoch': ep,
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, best_model_path)

        star = " * (BEST)" if is_best else ""
        print(f"Epoch {ep:2d}/{epochs} [{dt:.1f}s] - Train Loss: {loss:.4f} Acc: {acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% Macro-F1: {val_f1:.4f}{star}", flush=True)

    print(f"\nTraining Complete. Best Val Acc: {best_val_acc*100:.2f}%, Best Val Macro-F1: {best_val_f1:.4f}")

    # Evaluate Best Model on Test Set (PrivateTest)
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET (PrivateTest):")
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    cm_path = config.get('cm_path', 'reports/confusion_matrix.png')
    test_metrics = evaluate_model(model, test_loader, device=device, save_cm_path=cm_path)
    print_evaluation_summary(test_metrics)

    # Export best model to ONNX
    onnx_path = config.get('onnx_path', 'models/emotion_model.onnx')
    model.export_onnx(onnx_path, img_size=img_size)

    return {
        'val_accuracy': best_val_acc,
        'val_macro_f1': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'per_class': test_metrics['report']
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=112)
    parser.add_argument('--lr-backbone', type=float, default=3e-4)
    parser.add_argument('--lr-head', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    args = parser.parse_args()

    cfg = {
        'backbone': args.backbone,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'lr_backbone': args.lr_backbone,
        'lr_head': args.lr_head,
        'label_smoothing': args.label_smoothing,
        'use_class_weights': not args.no_class_weights,
        'resume_from': args.resume_from,
        'threads': 8,
        'save_path': 'models/best_emotion_model.pth',
        'onnx_path': 'models/emotion_model.onnx',
        'cm_path': 'reports/confusion_matrix.png'
    }
    run_training(cfg)
