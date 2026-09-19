import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from src.dataset import get_dataloaders, EMOTION_CLASSES
from src.model import build_model

def evaluate_model(model, dataloader, device='cpu', save_cm_path='reports/confusion_matrix.png'):
    """
    Evaluates model on dataloader.
    Returns: accuracy, macro_f1, per_class_metrics, cm
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    report_dict = classification_report(all_targets, all_preds, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    if save_cm_path:
        os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
        plt.title(f'FER2013 Confusion Matrix (Acc: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f})')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix plot to: {save_cm_path}")

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'report': report_dict,
        'confusion_matrix': cm
    }

def print_evaluation_summary(metrics: dict):
    acc = metrics['accuracy']
    macro_f1 = metrics['macro_f1']
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: Accuracy = {acc*100:.2f}% | Macro-F1 = {macro_f1:.4f}")
    print("="*60)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-"*60)
    for emotion in EMOTION_CLASSES:
        m = metrics['report'].get(emotion, {})
        print(f"{emotion:<12} | {m.get('precision', 0.0):<10.4f} | {m.get('recall', 0.0):<10.4f} | {m.get('f1-score', 0.0):<10.4f} | {int(m.get('support', 0)):<8}")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/best_emotion_model.pth')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small')
    parser.add_argument('--img-size', type=int, default=112)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--cm-path', type=str, default='reports/confusion_matrix.png')
    args = parser.parse_args()

    _, val_loader, test_loader, _ = get_dataloaders(img_size=args.img_size, batch_size=64)
    loader = test_loader if args.split == 'test' else val_loader

    model = build_model(backbone=args.backbone, num_classes=7, pretrained=False)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.model_path}")
    else:
        print(f"Model path {args.model_path} not found. Running with random weights.")

    metrics = evaluate_model(model, loader, device='cpu', save_cm_path=args.cm_path)
    print_evaluation_summary(metrics)
