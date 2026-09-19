import torch
import torch.nn as nn
import torchvision.models as models

class EmotionClassifier(nn.Module):
    """
    Lightweight Transfer Learning Classifier for 7 Facial Emotion Classes.
    Backbones supported: 'mobilenet_v3_small', 'mobilenet_v2', 'resnet18'
    """
    def __init__(self, backbone: str = 'mobilenet_v3_small', num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.pool = base.avgpool
            in_features = base.classifier[0].in_features
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=dropout / 2),
                nn.Linear(256, num_classes)
            )
        elif backbone == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = base.last_channel
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 256),
                nn.ReLU6(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(256, num_classes)
            )
        elif backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.pool = base.avgpool
            in_features = base.fc.in_features
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze feature extraction layers for head warm-up."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_all: bool = True):
        """Unfreeze backbone layers for fine-tuning."""
        if unfreeze_all:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze top layers only
            params = list(self.features.parameters())
            cutoff = int(len(params) * 0.5)
            for p in params[:cutoff]:
                p.requires_grad = False
            for p in params[cutoff:]:
                p.requires_grad = True

    def export_onnx(self, output_path: str, img_size: int = 112, opset_version: int = 17):
        """Export model to ONNX format with dynamic batch dimension."""
        self.eval()
        dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
        try:
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
                dynamo=False
            )
        except TypeError:
            # Older torch without dynamo keyword
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
            )
        print(f"Exported model successfully to ONNX: {output_path}")

def build_model(backbone: str = 'mobilenet_v3_small', num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
    return EmotionClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained, dropout=dropout)

if __name__ == '__main__':
    model = build_model('mobilenet_v3_small')
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    print("MobileNetV3-Small test forward output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
