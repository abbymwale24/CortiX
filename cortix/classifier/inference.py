"""
CortiX Module 3 — Production Inference Wrapper

Runs low-latency production inference by managing sliding sequences 
of live flow features, scaling, and forward pass classification.
"""

import os
import pickle
import logging
from collections import deque
import torch
import numpy as np

from cortix.config import config
from cortix.classifier.model import CortixLSTMCNN

logger = logging.getLogger("cortix.classifier.inference")


class ClassifierInference:
    """
    Production interface for the LSTM-CNN Classifier.
    
    Maintains a rolling window of recent flows to construct 
    temporal sequences of shape (10, 40) for realtime classification.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        seq_len: int | None = None,
        num_features: int | None = None,
    ):
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len or config.CLASSIFIER_SEQ_LEN
        self.num_features = num_features or config.CLASSIFIER_NUM_FEATURES

        # Rolling buffer for flow history per source/destination context
        # Key: src_ip, Value: deque of flow feature vectors
        self._flow_buffers: dict[str, deque] = {}

        # Load scalers and label encoders
        self.scaler = None
        self.encoder = None
        self.model = None

        self._load_assets()

    def _load_assets(self):
        """Load pretrained model, scaler, and label encoder."""
        try:
            scaler_path = "models/scaler.pkl"
            encoder_path = "models/label_encoder.pkl"

            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                logger.warning("No scaler found at %s. Inference will use dummy scaling.", scaler_path)

            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
            else:
                logger.warning("No label encoder found at %s.", encoder_path)

            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=torch.device(self.device))
                if isinstance(ckpt, dict) and "fc2.bias" in ckpt:
                    num_classes = ckpt["fc2.bias"].shape[0]
                elif self.encoder:
                    num_classes = len(self.encoder.classes_)
                else:
                    num_classes = config.CLASSIFIER_NUM_CLASSES

                self.model = CortixLSTMCNN(num_classes=num_classes)
                self.model.load_state_dict(ckpt)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Pretrained LSTM-CNN loaded onto %s", self.device)
            else:
                logger.warning("No model weights found at %s. Inference will return mock predictions.", self.model_path)
        except Exception as exc:
            logger.error("Failed to load inference assets: %s", exc)

    def predict_flow(self, src_ip: str, flow_vector: np.ndarray) -> dict:
        """
        Ingest a new flow vector, scale it, and classify it.
        
        Args:
            src_ip: IP address identifying the source
            flow_vector: 40-dimensional feature array (as defined in SELECTED_FEATURES)
            
        Returns:
            dict containing class, confidence, and alert flag
        """
        # Maintain rolling window sequence per IP
        if src_ip not in self._flow_buffers:
            self._flow_buffers[src_ip] = deque(maxlen=self.seq_len)
            
        self._flow_buffers[src_ip].append(flow_vector)

        # Return benign if buffer is not fully populated yet
        if len(self._flow_buffers[src_ip]) < self.seq_len:
            return {
                "class": "BENIGN",
                "confidence": 1.0,
                "is_threat": False,
                "status": "warming_up",
            }

        # Sequence input of shape (seq_len, num_features)
        seq_features = np.array(list(self._flow_buffers[src_ip]), dtype=np.float32)

        # Scale sequence
        if self.scaler:
            # Flatten to scale, then reshape back
            scaled_flat = self.scaler.transform(seq_features)
            seq_features = scaled_flat

        # Run inference if model exists
        if self.model:
            # Add batch dimension: (1, seq_len, num_features)
            tensor_in = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            if self.encoder:
                class_label = self.encoder.classes_[pred_idx]
            else:
                class_label = f"CLASS_{pred_idx}"
        else:
            # Mock fallback
            class_label = "BENIGN"
            confidence = 1.0

        is_threat = class_label != "BENIGN" and confidence >= config.CLASSIFIER_CONFIDENCE_THRESHOLD

        return {
            "class": class_label,
            "confidence": confidence,
            "is_threat": is_threat,
            "status": "active",
        }

    def clear_buffer(self, src_ip: str):
        """Clear historical sequence buffer for a specific host."""
        if src_ip in self._flow_buffers:
            self._flow_buffers[src_ip].clear()
