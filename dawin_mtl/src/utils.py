import os

import torch
import pickle
import math

import numpy as np
from scipy.special import digamma, gammaln, psi
from scipy.stats import dirichlet

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


import time
import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

class DirichletMixtureModel:
    """
    Dirichlet Mixture Model for data with each row in X 
    as a probability vector (summing to 1).
    """

    def __init__(self, n_components=3, random_seed=1):
        self.n_mixtures = n_components
        self.random_seed = random_seed
        self.convergence = False


    def _init_clusters(self, X, init_idx):
        """Initialize mixture responsibilities via k-means or random approach."""
        N = self.n_observations
        if self.method == "kmeans":
            kmeans = KMeans(
                n_clusters=self.n_components,
                n_init=1,
                random_state=self.random_seed + init_idx
            )
            labels = kmeans.fit(X).labels_
            resp = np.zeros((N, self.n_components))
            resp[np.arange(N), labels] = 1.0
        else:  # random
            rng = np.random.default_rng(self.random_seed + init_idx)
            resp = rng.random((N, self.n_components))
            resp /= resp.sum(axis=1, keepdims=True)

        # Small stability offset
        resp += 1e-14
        
        # Random initial Dirichlet parameters
        self.dirichlet_params_ = np.random.rand(self.n_components, self.n_components)

        # Single M-step to set initial params
        self._M_step(X, np.log(resp))

    def _estimate_log_weights(self):
        """Return log of current mixture weights."""
        return np.log(self.weights_)

    def _dirichlet_log_prob_single(self, X, mixture_idx):
        """Compute log-prob for one mixture across all samples."""
        alpha = self.dirichlet_params_[mixture_idx]
        log_C = gammaln(alpha.sum()) - np.sum(gammaln(alpha))
        return log_C + np.sum((alpha - 1) * np.log(X), axis=1)

    def _estimate_log_prob(self, X):
        """Compute log-prob of each mixture for each sample."""
        log_prob = np.empty((self.n_observations, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = self._dirichlet_log_prob_single(X, k)

        return log_prob

    def _weighted_log_prob(self, X):
        """Return mixture-weighted log-prob."""
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_resp(self, X):
        """
        E-step: 
        log_resp = log p(x | mixture) + log(weights) 
                   - logsumexp over mixtures
        """
        wlp = self._weighted_log_prob(X)
        log_norm = logsumexp(wlp, axis=1)
        with np.errstate(under="ignore"):
            log_resp = wlp - log_norm[:, None]
        return log_norm, log_resp

    def _compute_responsibilities(self, log_resp):
        """Exponentiate log responsibilities and sum them per mixture."""
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 1e-14
        return resp, nk

    def _update_weights(self, nk):
        """Update mixture weights."""
        self.weights_ = nk / nk.sum()

    def _E_step(self, X):
        """Perform E-step, returning mean log-likelihood and log responsibilities."""
        log_norm, log_resp = self._estimate_log_resp(X)
        return log_norm.mean(), log_resp

    def _M_step(self, X, log_resp):
        """Perform M-step using updated responsibilities."""
        resp, nk = self._compute_responsibilities(log_resp)
        self._update_weights(nk)

        # Update each mixture's Dirichlet parameter
        #! Note that this is a simplified version with moment-matching heuristic, rather than the exact EM
        for k in range(self.n_components):
            alpha_new = resp[:, k] @ X  # Weighted sum of X
            self.dirichlet_params_[k] = alpha_new / nk[k]

    def fit(self, X, n_init=10, method="kmeans", max_iter=2000, tol=1e-6):
        """
        Run EM to fit a Dirichlet mixture model to data X.
        Each row of X should be a probability vector (sum to 1).
        """
        self.n_observations, self.n_components = X.shape
        self.method = method
        self.convergence = False

        best_lower_bound = -np.inf
        best_params = None

        for init_idx in range(n_init):
            print(f"{init_idx + 1}-th DMM initialization")
            self._init_clusters(X, init_idx)

            lower_bound = -np.inf

            for iteration in range(max_iter):
                prev_bound = lower_bound

                log_prob_norm, log_resp = self._E_step(X)
                self._M_step(X, log_resp)

                lower_bound = log_prob_norm
                diff = lower_bound - prev_bound

                if abs(diff) < tol:
                    self.convergence = True
                    break

            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                _, nk = self._compute_responsibilities(log_resp)
                self._update_weights(nk)
                best_params = (self.weights_.copy(), self.dirichlet_params_.copy())

        # Restore the best parameters
        self.weights_, self.params_ = best_params
        self.lower_bound_ = best_lower_bound
        return self

    def predict_proba(self, X):
        """Return soft assignments (N x K)."""
        _, log_resp = self._estimate_log_resp(X)
        return np.exp(log_resp)

    def predict(self, X):
        """Return the most likely mixture for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)