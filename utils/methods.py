import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from functools import partial

def g(logits, temperature=1.0):
    return torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1)


def doctor(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    g_out = g(logits=logits, temperature=temperature)
    return (1 - g_out) / g_out


def odin(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    return -torch.softmax(logits / temperature, dim=1).max(dim=1)[0]


def msp(logits: torch.Tensor, **kwargs):
    return -torch.softmax(logits, dim=1).amax(dim=1)


def entropy(logits: torch.Tensor, **kwargs):
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def logitgap(logits, temperature=1.0, **kwargs):
    B, C = logits.shape
    # descending order
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    # anchor: k-th largest logit
    anchor = sorted_logits[:, :1]
    competitors = sorted_logits[:, 1:]
    score = anchor.squeeze(1) - competitors.mean(dim=1)
    return -score


def logitgapN(logits, k=1, topn=1, temperature=1.0, **kwargs):
    B, C = logits.shape
    # descending order
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    # anchor: k-th largest logit
    anchor = sorted_logits[:, k-1:k]
    if topn > 0:
        competitors = sorted_logits[:, k:k+topn]
        score = anchor.squeeze(1) - competitors.mean(dim=1)
    else:
        score = anchor.squeeze(1)
    return -score


def cmd(logits, temperature=1.0, tau=0.5, **kwargs):
    B, C = logits.shape
    logits = logits / temperature
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    margins = sorted_logits[:, :1] - sorted_logits[:, 1:]
    margins = margins / tau
    # non-saturating transform
    dom = torch.log1p(margins)
    # instability
    instability = dom.std(dim=1)
    score = instability 
    return score


class MetricLearningLagrange:
    def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.lbd = lbd
        self.temperature = temperature
        self.params = None

    def fit(self, train_dataloader, *args, **kwargs):
        # get train logits
        train_logits = []
        train_labels = []
        for data, labels in tqdm(train_dataloader, desc="Fitting metric"):
            data = data.to(self.device)
            with torch.no_grad():
                logits = self.model(data).cpu()
            if logits.shape[1] % 2 == 1:  # openmix
                logits = logits[:, :-1]
            train_logits.append(logits)
            train_labels.append(labels)
        train_logits = torch.cat(train_logits, dim=0)
        train_pred = train_logits.argmax(dim=1)
        train_labels = torch.cat(train_labels, dim=0)
        train_labels = (train_labels != train_pred).int()

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)
        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

        self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
        self.params = torch.tril(self.params, diagonal=-1)
        self.params = self.params + self.params.T
        self.params = torch.relu(self.params)
        if torch.all(self.params <= 0):
            # default to gini
            self.params = torch.ones(self.params.size()).to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
        self.params = self.params / self.params.norm()

    def __call__(self, logits, *args, **kwds):
        probs = torch.softmax(logits / self.temperature, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)

    def export_matrix(self):
        return self.params.cpu()


class Wrapper:
    def __init__(self, method, *args, **kwargs):
        self.method = method

    def fit(self, train_dataloader, val_dataloader, **kwargs):
        if hasattr(self.method, "fit"):
            self.method.fit(train_dataloader, val_dataloader, **kwargs)
        return self

    def __call__(self, x):
        return self.method(x)

    def export_matrix(self):
        return self.method.export_matrix()


def get_method(method_name, *args, **kwargs):
    if method_name == "doctor":
        return Wrapper(partial(doctor, *args, **kwargs))
    if method_name == "odin":
        return Wrapper(partial(odin, *args, **kwargs))
    if method_name == "msp":
        return Wrapper(msp)
    if method_name == "logitgap":
        return Wrapper(partial(logitgap, *args, **kwargs))
    if method_name == "logitgapN":
        return Wrapper(partial(logitgapN, *args, **kwargs))
    if method_name == "cmd":
        return Wrapper(partial(cmd, *args, **kwargs))
    if method_name == "relu":
        return Wrapper(MetricLearningLagrange(*args, **kwargs))
    if method_name == "entropy":
        return Wrapper(entropy)
    raise ValueError(f"Method {method_name} not supported")











