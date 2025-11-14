# method/dann.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer
    Forward: identity
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """
    Patient discriminator for DANN
    Try to predict which patient the features came from
    """
    def __init__(self, feature_dim=768, num_patients=126, hidden_dim=256):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 2, num_patients)
        )
    
    def forward(self, x):
        return self.discriminator(x)


class PAFAWithDANN(nn.Module):
    """
    PAFA with DANN
    
    Components:
    1. PCSL: Patient-level contrastive (unchanged)
    2. DANN: Domain adversarial (replacing GPAL)
    """
    def __init__(self, feature_dim=768, num_patients=126, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.feature_dim = feature_dim
        self.num_patients = num_patients
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            num_patients=num_patients
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # GRL lambda scheduler
        self.grl_lambda = 0.0
        self.epoch = 0
        self.max_epochs = 100
    
    def update_grl_lambda(self, epoch, max_epochs=None):
        """
        Update GRL lambda with schedule
        Standard DANN practice: gradually increase from 0 to 1
        """
        if max_epochs is not None:
            self.max_epochs = max_epochs
        
        self.epoch = epoch
        
        # Schedule: λ = 2/(1+exp(-10*p)) - 1, where p = epoch/max_epochs
        p = float(epoch) / float(self.max_epochs)
        self.grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1.
        self.grl.lambda_ = self.grl_lambda
    
    def compute_pcsl_loss(self, features, patient_ids):
        """
        PCSL: Patient-level Contrastive Loss (unchanged)
        Goal: Minimize within-patient variance, maximize between-patient distance
        """
        device = features.device
        unique_ids = torch.unique(patient_ids)
        
        if unique_ids.numel() <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        within_variance = 0.0
        centroids = []
        
        # For each patient
        for pid in unique_ids:
            mask = (patient_ids == pid)
            features_pid = features[mask]
            centroid = features_pid.mean(dim=0)
            centroids.append(centroid)
            within_variance += torch.sum((features_pid - centroid) ** 2)
        
        centroids = torch.stack(centroids, dim=0)
        
        # Between-patient distance
        between_distance = 0.0
        num_patients = centroids.shape[0]
        for i in range(num_patients):
            for j in range(i + 1, num_patients):
                between_distance += torch.norm(centroids[i] - centroids[j]) ** 2
        
        # PCSL loss
        loss_pcsl = within_variance / (between_distance + self.eps)
        
        return loss_pcsl
    
    def compute_dann_loss(self, features, patient_ids):
        """
        DANN: Domain Adversarial Loss (replacing GPAL)
        Goal: Learn patient-invariant features via adversarial training
        """
        device = features.device
        
        # Check if we have multiple patients
        unique_ids = torch.unique(patient_ids)
        if unique_ids.numel() <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Map patient IDs to continuous indices [0, N-1]
        patient_id_mapping = {pid.item(): idx for idx, pid in enumerate(unique_ids)}
        mapped_ids = torch.tensor(
            [patient_id_mapping[pid.item()] for pid in patient_ids],
            device=device,
            dtype=torch.long
        )
        
        # Apply gradient reversal
        reversed_features = self.grl(features)
        
        # Domain discrimination
        domain_logits = self.domain_discriminator(reversed_features)
        
        # Cross-entropy loss
        # Discriminator tries to classify patient correctly
        # But gradient is reversed, so feature extractor learns to confuse it
        loss_dann = F.cross_entropy(domain_logits, mapped_ids)
        
        return loss_dann
    
    def forward(self, features, patient_ids, lambda_pcsl=50.0, lambda_dann=0.5):
        """
        Forward pass
        
        Args:
            features: [B, N, D] or [B, D] features
            patient_ids: [B] patient IDs
            lambda_pcsl: weight for PCSL loss
            lambda_dann: weight for DANN loss (replacing lambda_gpal)
        
        Returns:
            total_loss: weighted combination
        """
        # Handle [B, N, D] format - take mean over sequence
        if features.dim() == 3:
            features = features.mean(dim=1)  # [B, D]
        
        # Compute PCSL loss (unchanged)
        loss_pcsl = self.compute_pcsl_loss(features, patient_ids)
        
        # Compute DANN loss (replacing GPAL)
        loss_dann = self.compute_dann_loss(features, patient_ids)
        
        # Total loss
        total_loss = lambda_pcsl * loss_pcsl + lambda_dann * loss_dann
        
        return total_loss