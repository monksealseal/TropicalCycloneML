"""PyTorch neural network for tropical cyclone intensity prediction.

CycloneIntensityNet is a CNN-LSTM hybrid that predicts future cyclone intensity
from satellite imagery sequences and environmental predictors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class SatelliteEncoder(nn.Module):
        """CNN encoder for satellite imagery patches (single-channel IR or multi-channel)."""

        def __init__(self, in_channels: int = 1, feature_dim: int = 128):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(256 * 4 * 4, feature_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode a satellite image to a feature vector.

            Args:
                x: Image tensor of shape (B, C, H, W).

            Returns:
                Feature tensor of shape (B, feature_dim).
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class CycloneIntensityNet(nn.Module):
        """CNN-LSTM model for tropical cyclone intensity prediction.

        Architecture:
        1. CNN encoder processes satellite imagery at each timestep
        2. Environmental predictors are concatenated with image features
        3. LSTM processes the temporal sequence
        4. FC head predicts intensity at multiple forecast lead times

        Input:
            - Satellite image sequence: (B, T, C, H, W)
            - Environmental predictors: (B, T, N_env)
              N_env features: SST, wind_shear, RH_700, CAPE, lat, lon, current_wind, pressure

        Output:
            - Predicted wind speeds at 5 lead times: (B, 5)
              [12h, 24h, 48h, 72h, 120h]
        """

        def __init__(
            self,
            image_channels: int = 1,
            image_feature_dim: int = 128,
            env_features: int = 8,
            lstm_hidden: int = 256,
            lstm_layers: int = 2,
            forecast_steps: int = 5,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.image_encoder = SatelliteEncoder(image_channels, image_feature_dim)
            self.env_fc = nn.Linear(env_features, 32)

            combined_dim = image_feature_dim + 32
            self.lstm = nn.LSTM(
                input_size=combined_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
            )

            self.head = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, forecast_steps),
            )

            # Output is wind speed in knots (always positive)
            self.forecast_steps = forecast_steps

        def forward(
            self,
            images: torch.Tensor,
            env_predictors: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                images: Satellite imagery sequence (B, T, C, H, W).
                env_predictors: Environmental predictors (B, T, N_env).

            Returns:
                Predicted wind speeds in knots (B, forecast_steps).
            """
            B, T, C, H, W = images.shape

            # Encode each image in the sequence
            images_flat = images.view(B * T, C, H, W)
            img_features = self.image_encoder(images_flat)  # (B*T, feature_dim)
            img_features = img_features.view(B, T, -1)  # (B, T, feature_dim)

            # Process environmental predictors
            env_features = F.relu(self.env_fc(env_predictors))  # (B, T, 32)

            # Combine and run through LSTM
            combined = torch.cat([img_features, env_features], dim=-1)  # (B, T, combined_dim)
            lstm_out, _ = self.lstm(combined)  # (B, T, lstm_hidden)

            # Use final timestep for prediction
            final_state = lstm_out[:, -1, :]  # (B, lstm_hidden)

            # Predict intensity (use softplus to ensure positive output)
            predictions = F.softplus(self.head(final_state))  # (B, forecast_steps)

            return predictions

    def create_model(
        image_channels: int = 1,
        pretrained: bool = False,
    ) -> CycloneIntensityNet:
        """Create a CycloneIntensityNet model.

        Args:
            image_channels: Number of channels in satellite imagery.
            pretrained: If True, load pretrained weights (not yet available).

        Returns:
            Initialized model.
        """
        model = CycloneIntensityNet(image_channels=image_channels)
        if pretrained:
            # Placeholder for future pretrained weights
            pass
        return model

else:
    # Stub when PyTorch is not available
    def create_model(**kwargs: Any) -> None:
        raise ImportError("PyTorch is required for the CycloneIntensityNet model. "
                         "Install with: pip install torch")


def get_model_summary() -> dict[str, Any]:
    """Get a summary of the CycloneIntensityNet architecture."""
    if not HAS_TORCH:
        return {"error": "PyTorch not available", "install": "pip install torch"}

    model = CycloneIntensityNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": "CycloneIntensityNet",
        "architecture": "CNN-LSTM Hybrid",
        "components": {
            "image_encoder": "4-layer CNN with BatchNorm (SatelliteEncoder)",
            "environmental_encoder": "Linear projection (8 -> 32)",
            "temporal_model": "2-layer LSTM (hidden=256)",
            "prediction_head": "3-layer MLP",
        },
        "input": {
            "satellite_images": "(batch, timesteps, channels, 256, 256)",
            "environmental_predictors": "(batch, timesteps, 8)",
            "env_features": [
                "SST", "wind_shear", "RH_700hPa", "CAPE",
                "latitude", "longitude", "current_wind_kt", "pressure_hPa",
            ],
        },
        "output": {
            "shape": "(batch, 5)",
            "description": "Predicted max sustained wind (kt) at 12h, 24h, 48h, 72h, 120h",
        },
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_count_human": f"{total_params / 1e6:.1f}M",
    }
