"""
Attention Based MIL
reference: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class AttentionBase(nn.Module):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super(AttentionBase, self).__init__()
        # parameters
        self.M = n_features
        self.L = hidden_layer
        self.ATTENTION_BRANCHES = attention_branches
        self.label_smoothing=label_smoothing
        # attention
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_labels),
            nn.Sigmoid()
        )

    def forward_single(self, X):
        """for single WSI (instances x features)"""
        A = self.calc_attention(X, [1,0])  # KxATTENTION_BRANCHES
        Z = torch.mm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        Y_hat = (Y_prob>.5).float()
        return Y_prob, Y_hat, A

    def forward_multiwsi(self, X):
        """for Multi WSI (batch x instances x features)"""
        A = self.calc_attention(X, [2,1])  # KxATTENTION_BRANCHES
        Z = torch.bmm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        Y_hat = (Y_prob>.5).float()
        return Y_prob, Y_hat, A

    def calc_attention(self, X, position):
        A = self.attention(X)
        A = torch.transpose(A, *position)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=position[0])  # softmax over K
        return A

    def calc_loss(self, X, Y):
        Y = Y.float()
        Y_prob, _, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood

    def calc_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calc_error_and_loss(self, X, Y):
        Y=Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        # acc
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        # loss
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return error, neg_log_likelihood

    def predict_proba(self, X):
        return torch.squeeze(self.forward(X)[0].detach()).cpu().numpy()

class GatedAttentionBase(nn.Module):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super(GatedAttentionBase, self).__init__()
        # parameters
        self.M = n_features
        self.L = hidden_layer
        self.ATTENTION_BRANCHES = attention_branches
        self.label_smoothing=label_smoothing
        # attention
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_labels),
            nn.Sigmoid()
        )

    def forward_single(self, X):
        """for single WSI (instances x features)"""
        A = self.calc_attention(X, [1,0])
        Z = torch.mm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        Y_hat = (Y_prob>.5).float()
        return Y_prob, Y_hat, A

    def forward_multiwsi(self, X):
        """for Multi WSI (batch x instances x features)"""
        A = self.calc_attention(X, [2,1])
        Z = torch.bmm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        Y_hat = (Y_prob>.5).float()
        return Y_prob, Y_hat, A

    def calc_attention(self, X, position):
        A_V = self.attention_V(X)  # KxL
        A_U = self.attention_U(X)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, *position)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=position[0])  # softmax over K
        return A

    def calc_loss(self, X, Y):
        Y = Y.float()
        Y_prob, _, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood

    def calc_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calc_error_and_loss(self, X, Y):
        Y=Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        # acc
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        # loss
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return error, neg_log_likelihood
        
    def predict_proba(self, X):
        return torch.squeeze(self.forward(X)[0].detach()).cpu().numpy()

class AttentionWSI(AttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing)

    def forward(self, X):
        return self.forward_single(X)

class AttentionMultiWSI(AttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing)

    def forward(self, X):
        return self.forward_multiwsi(X)

class GatedAttentionWSI(GatedAttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing)

    def forward(self, X):
        return self.forward_single(X)

class GatedAttentionMultiWSI(GatedAttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None,):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing)

    def forward(self, X):
        return self.forward_multiwsi(X)
