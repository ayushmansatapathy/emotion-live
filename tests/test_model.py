import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import build_model, EmotionClassifier

def test_model_forward_shape():
    model = build_model('mobilenet_v3_small', num_classes=7, pretrained=False)
    model.eval()
    dummy_input = torch.randn(4, 3, 112, 112)
    output = model(dummy_input)
    assert output.shape == (4, 7), f"Expected shape (4, 7), got {output.shape}"

def test_model_freezing_unfreezing():
    model = build_model('mobilenet_v3_small', num_classes=7, pretrained=False)
    model.freeze_backbone()
    # Check backbone parameters require_grad is False
    for p in model.features.parameters():
        assert not p.requires_grad
    # Classifier head parameters should still require grad
    for p in model.classifier.parameters():
        assert p.requires_grad

    model.unfreeze_backbone(unfreeze_all=True)
    for p in model.features.parameters():
        assert p.requires_grad

def test_model_mobilenet_v2():
    model = build_model('mobilenet_v2', num_classes=7, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 112, 112)
    output = model(dummy_input)
    assert output.shape == (2, 7)
