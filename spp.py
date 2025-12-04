import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack

class SPP(Attack):
    def __init__(self, model, eps=8/255, steps=5,alpha=2/255):
        super().__init__('SPP',model)
        self.model = model
        self.eps = eps
        self.steps = steps
        self.device = next(model.parameters()).device
        self.alpha = alpha
    def forward(self, images,labels):

        self.model.eval()
        B,C,H,W = images.shape
        ori_outputs = self.model(images).detach()
        adv_images = images.clone().detach().to(self.device)
        adv_images = adv_images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0, 1)
        criterion = nn.KLDivLoss(reduction='sum')
        for step in range(self.steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            loss_ce = F.cross_entropy(outputs,labels)
            loss_kl = criterion(F.log_softmax(outputs, dim=-1), F.softmax(ori_outputs, dim=-1))
            loss = loss_kl + loss_ce
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.alpha * grad.view(B, C, H, W).sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images




