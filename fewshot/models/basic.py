import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from fewshot.models.model_factory import RegisterModel
from fewshot.models.utils import *
from fewshot.data.episode import Episode


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


@RegisterModel("protonet")
class Protonet(nn.Module):

    def __init__(self, config, dataset):
        super().__init__()

        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ─────────────────────────────────────────────
        # Learned cluster variances (log-space)
        # ─────────────────────────────────────────────
        self.log_sigma_l = nn.Parameter(
            torch.log(torch.tensor([config.init_sigma_l], dtype=torch.float32)),
            requires_grad=config.learn_sigma_l
        )

        self.log_sigma_u = nn.Parameter(
            torch.log(torch.tensor([config.init_sigma_u], dtype=torch.float32)),
            requires_grad=config.learn_sigma_u
        )

        # ─────────────────────────────────────────────
        # Encoder (Conv-4 backbone)
        # ─────────────────────────────────────────────
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
            conv_block(config.num_channel, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            Flatten()
        )

        self._init_weights()
        self.to(self.device)

    # ─────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    # ─────────────────────────────────────────────
    # Distance computation
    # ─────────────────────────────────────────────
    def _compute_distances(self, protos, example):
        """
        protos: [B, C, D]
        example: [D] or [B, D]
        """
        return ((example.unsqueeze(1) - protos) ** 2).sum(dim=2)

    # ─────────────────────────────────────────────
    # Encoder forward
    # ─────────────────────────────────────────────
    def _run_forward(self, x):
        """
        x: [B, N, C, H, W]
        returns: [B, N, D]
        """
        B, N = x.shape[:2]
        x = x.view(B * N, *x.shape[2:])
        h = self.encoder(x)
        return h.view(B, N, -1)

    # ─────────────────────────────────────────────
    # Prototype computation (soft assignments)
    # ─────────────────────────────────────────────
    def _compute_protos(self, h, probs):
        """
        h:     [B, N, D]
        probs: [B, N, C]
        """
        probs = probs.unsqueeze(-1)              # [B, N, C, 1]
        h = h.unsqueeze(2)                       # [B, N, 1, D]

        weighted_sum = (h * probs).sum(dim=1)    # [B, C, D]
        counts = probs.sum(dim=1)                # [B, C, 1]

        # numerical safety
        counts = counts.clamp_min(1e-8)
        return weighted_sum / counts

    # ─────────────────────────────────────────────
    # Episodic batch processing
    # ─────────────────────────────────────────────
    def _process_batch(self, batch, super_classes=False):

        def to_tensor(x):
            return torch.from_numpy(x).float().to(self.device)

        x_train = to_tensor(batch.x_train)
        x_test = to_tensor(batch.x_test)

        if batch.x_unlabel is not None and batch.x_unlabel.size > 0:
            x_unlabel = to_tensor(batch.x_unlabel)
            y_unlabel = torch.from_numpy(batch.y_unlabel).long().to(self.device)
        else:
            x_unlabel, y_unlabel = None, None

        if super_classes:
            y_train = torch.from_numpy(batch.y_train_str[:, 1]).long().unsqueeze(0)
            y_test = torch.from_numpy(batch.y_test_str[:, 1]).long().unsqueeze(0)
        else:
            y_train = torch.from_numpy(batch.y_train[:, :, 1]).long()
            y_test = torch.from_numpy(batch.y_test[:, :, 1]).long()

        y_train = y_train.to(self.device)
        y_test = y_test.to(self.device)

        return Episode(
            x_train=x_train,
            y_train=y_train,
            train_indices=np.expand_dims(batch.train_indices, 0),
            x_test=x_test,
            y_test=y_test,
            test_indices=np.expand_dims(batch.test_indices, 0),
            x_unlabel=x_unlabel,
            y_unlabel=y_unlabel,
            unlabel_indices=np.expand_dims(batch.unlabel_indices, 0),
            y_train_str=batch.y_train_str,
            y_test_str=batch.y_test_str
        )

    # ─────────────────────────────────────────────
    # Abstract forward
    # ─────────────────────────────────────────────
    def forward(self, sample):
        raise NotImplementedError("Use a subclass (e.g. IMPModel)")