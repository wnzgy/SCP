import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack
import cupy as cp

class SCP(Attack):
    def __init__(self, model, eps=8/255, steps=5,alpha=2/255):
        super().__init__('SCP',model)
        self.model = model
        self.eps = eps
        self.steps = steps
        self.device = next(model.parameters()).device
        self.alpha = alpha
    def forward(self, images,labels):
        self.model.eval()
        B,C,H,W = images.shape
        Q, prob = self.compute_prob_and_gradients_batch(self.model, images.clone())
        adv_images = images.clone().detach().to(self.device)
        adv_images = adv_images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0, 1)
        eigenvectors_A, eigenvalues_B = self.compute_princial_eigenvector_batch(Q, prob)
        eigen = eigenvectors_A[-1]

        for step in range(self.steps):
            adv_images.requires_grad_(True)
            adv_images = adv_images.detach() + self.alpha * eigen.view(B, C, H, W).sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def compute_prob_and_gradients_batch(self,model, x_batch):
        model = model.to(self.device)
        x_batch = x_batch.to(self.device).requires_grad_(True)
        B = x_batch.shape[0]
        d = int(x_batch.numel() / B)
        logits = model(x_batch)  # (B x K)
        probs = F.softmax(logits, dim=1)  # (B x K)

        num_classes = probs.shape[1]

        Q = torch.zeros((B, d, num_classes), device=self.device)


        for k in range(num_classes):

            log_probs = torch.log(probs[:, k]+1e-9)
            gradients = torch.autograd.grad(
                log_probs.sum(), x_batch, retain_graph=True, create_graph=False
            )[0]

            Q[:, :, k] = gradients.view(B, -1)  # (B x d)
        return Q, probs.detach()
    def compute_princial_eigenvector_batch(self,Q_batch, prob_batch,reg_term = 1e-20):
        B, d, m = Q_batch.shape
        device = Q_batch.device

        prob_sqrt = torch.sqrt(prob_batch)  # [B, m]
        Q_prob_sqrt = Q_batch * prob_sqrt.unsqueeze(1)  # [B, d, m]


        # [B,d,m] [B,d,m] -> [B,m,m]
        B_matrices = torch.einsum('bdi,bdj->bij', Q_prob_sqrt, Q_prob_sqrt)  # [B, m, m]

        eye_matrix = torch.eye(m, device=device).unsqueeze(0).expand(B, -1, -1)
        B_matrices = B_matrices + reg_term * eye_matrix

        B_matrices = cp.asarray(B_matrices.cpu().numpy())
        eigenvalues_B, eigenvectors_B = cp.linalg.eigh(B_matrices)

        eigenvalues_B = torch.tensor(eigenvalues_B, device=self.device)
        eigenvectors_B = torch.tensor(eigenvectors_B, device=self.device)

        principal_eigenvectors_A = torch.einsum(
            'bdm,bmk->bdk', Q_prob_sqrt, eigenvectors_B
        )
        principal_eigenvectors_A = principal_eigenvectors_A.permute(2,0,1)
        return principal_eigenvectors_A, eigenvalues_B


