import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaborFilter(nn.Module):
    def __init__(self, n_filters, kernel_size, sample_rate, stride=1, Q=12.0):
        super(GaborFilter, self).__init__()
        assert kernel_size % 2 == 1, "use an odd kernel_size for centered 'same' conv"
        self.sr = sample_rate
        self.stride = stride
        self.Q = Q                            # static quality factor: bw_hz = cf_hz / Q

        # init log-spaced across the birdsong band, not random
        init_hz = torch.logspace(math.log10(1000), math.log10(8000), n_filters)
        # store the logit so sigmoid() in forward recovers the intended fraction of Nyquist
        frac = (init_hz / (sample_rate/2)).clamp(1e-4, 1 - 1e-4)  # 8000 Hz == Nyquist -> clamp
        self.center_freq = nn.Parameter(torch.log(frac / (1 - frac)))
        # bandwidth is not learned: constant-Q, derived from center_freq in forward
        t = torch.arange(-(kernel_size//2), kernel_size//2 + 1) / sample_rate
        self.register_buffer('t', t)          # fixed grid, not learned

    def build_kernel(self):
        """Materialize the [2*n_filters, 1, K] conv weight from the learnable params.

        Factored out of forward so an export/inference path can *bake* the filters into a
        plain Conv1d weight (no runtime sin/cos/exp), which is what MCU runtimes need.
        """
        cf = torch.sigmoid(self.center_freq) * (self.sr / 2)   # keep < Nyquist
        bw_hz   = cf / self.Q                                  # constant-Q: bandwidth tracks center freq
        sigma_t = 1.0 / (2*math.pi*bw_hz)                      # = Q/(2*pi*cf); Hz bandwidth -> time sigma
        gauss   = torch.exp(-0.5 * (self.t[None,:] / sigma_t[:,None])**2)
        # cos + sin quadrature pair for phase-invariant energy
        arg     = 2*math.pi*cf[:,None]*self.t[None,:]
        cos_k   = gauss * torch.cos(arg)
        sin_k   = gauss * torch.sin(arg)
        kernel  = torch.stack([cos_k, sin_k], dim=0)           # [2, n_filters, K]
        kernel  = kernel - kernel.mean(dim=-1, keepdim=True)   # remove DC leakage
        kernel  = kernel / kernel.norm(p=2, dim=-1, keepdim=True)
        return kernel.reshape(-1, 1, self.t.shape[-1])         # [2*n_filters, 1, K]

    def forward(self, x):                                       # x: [B, 1, L]
        kernel  = self.build_kernel()
        out     = F.conv1d(x, kernel, stride=self.stride, padding=self.t.shape[-1]//2)
        # combine quadrature pair -> magnitude: [B, n_filters, L']
        cos_out, sin_out = out.chunk(2, dim=1)
        return torch.sqrt(cos_out**2 + sin_out**2 + 1e-8)

    def output_lengths(self, lengths):
        """Map raw-waveform lengths to post-conv frame counts (matches forward's conv1d)."""
        K = self.t.shape[-1]
        return (lengths + 2*(K//2) - K) // self.stride + 1

class CNNHead(nn.Module):
    """
    Compact 2D-CNN head over the Gabor feature map [B, n_filters, T], treated as a
    single-channel time-frequency image. Three Conv-BN-ReLU blocks (each with 2x2
    max pooling) grow the receptive field and channel depth, then a *masked global
    max* pool keeps the head presence-sensitive -- a short call in a long clip can
    still drive the logit, unlike average pooling -- before a linear classifier.
    Returns [B] logits (train with BCEWithLogitsLoss).
    """
    def __init__(self, channels=(16, 32, 64), n_classes=1):
        super(CNNHead, self).__init__()
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                     # halve both freq and time each block
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.out_channels = in_ch
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x, mask=None):                 # x: [B, n_filters, T]
        x = torch.log1p(x).unsqueeze(1)              # [B, 1, F, T] single-channel image
        x = self.features(x)                         # [B, C, F', T'] (time downsampled ~8x)
        if mask is not None:
            # downsample the time mask to T' (a pooled column is valid if its window held
            # any valid frame), then blank padded columns with -inf so they can't win the max
            m = F.adaptive_max_pool1d(mask.float()[:, None, :], x.shape[-1]) > 0.5  # [B, 1, T']
            x = x.masked_fill(~m[:, :, None, :], float("-inf"))                     # broadcast over C, F'
        pooled = x.amax(dim=(2, 3))                  # [B, C] masked global max pool
        return self.fc(pooled).squeeze(-1)           # [B] logits (n_classes=1)

class LinearHead(nn.Module):
    """
    Minimal classifier head for the Gabor filterbank. Consumes [B, n_filters, T]:
    log-compress -> per-band norm -> logmeanexp pooling over time -> linear.
    logmeanexp is a soft-max over time (tau -> 0 approaches max, tau -> inf
    approaches mean), so a brief call in a long clip can still drive the logit --
    better than mean pooling for presence detection. Returns [B] raw logits
    (train with BCEWithLogitsLoss).
    """
    def __init__(self, n_filters, n_classes=1, tau=1.0):
        super(LinearHead, self).__init__()
        self.tau = tau
        self.bn = nn.BatchNorm1d(n_filters)          # per-band mean/var norm (loudness invariance)
        self.fc = nn.Linear(n_filters, n_classes)

    def forward(self, x, mask=None):                 # x: [B, n_filters, T], mask: [B, T] (True=valid)
        x = torch.log1p(x)                           # log compression
        x = self.bn(x)
        if mask is not None:
            # exclude padded frames from the pool: -inf drops out of logsumexp exactly
            x = x.masked_fill(~mask[:, None, :], float('-inf'))
            denom = mask.sum(-1).clamp(min=1).log()[:, None]   # log(valid frames) per sample
        else:
            denom = math.log(x.shape[-1])
        pooled = self.tau * (torch.logsumexp(x / self.tau, dim=-1) - denom)  # logmeanexp -> [B, n_filters]
        return self.fc(pooled).squeeze(-1)           # [B] logits (n_classes=1)

class GaborNet(nn.Module):
    """
    Learnable Gabor filterbank front-end + small 2D CNN head for clip-level
    binary classification (bird present / absent). Returns raw logits, so train
    with BCEWithLogitsLoss (not binary_cross_entropy).
    """
    def __init__(self, feature_extractor, head):
        super(GaborNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x, lengths=None):            # x: [B, 1, L], lengths: [B] valid samples
        feats = self.feature_extractor(x)          # [B, n_filters, T]
        mask = None
        if lengths is not None:
            out_len = self.feature_extractor.output_lengths(lengths.to(feats.device))
            T = feats.shape[-1]
            mask = torch.arange(T, device=feats.device)[None, :] < out_len[:, None]  # [B, T]
        return self.head(feats, mask)


def get_config(model):
    """Read the live architecture straight from the modules so it can be rebuilt exactly.

    Channel counts are read from the actual layers, not construction-time defaults, so a
    model that has been pruned by `train_with_slimming` round-trips correctly.
    """
    gf = model.feature_extractor
    cfg = {
        "feature_extractor": gf.__class__.__name__,
        "n_filters": int(gf.center_freq.shape[0]),
        "kernel_size": int(gf.t.shape[-1]),
        "sample_rate": int(gf.sr),
        "stride": int(gf.stride),
        "Q": float(gf.Q),
        "head": model.head.__class__.__name__,
    }
    head = model.head
    cfg["n_classes"] = int(head.fc.out_features)
    if isinstance(head, CNNHead):
        cfg["channels"] = [m.out_channels for m in head.features if isinstance(m, nn.Conv2d)]
    elif isinstance(head, LinearHead):
        cfg["tau"] = float(head.tau)
    return cfg


def build_from_config(cfg):
    """Reconstruct an (untrained) GaborNet with the architecture described by `cfg`."""
    gf = GaborFilter(cfg["n_filters"], cfg["kernel_size"], cfg["sample_rate"],
                     stride=cfg["stride"], Q=cfg["Q"])
    if cfg["head"] == "CNNHead":
        head = CNNHead(channels=tuple(cfg["channels"]), n_classes=cfg["n_classes"])
    elif cfg["head"] == "LinearHead":
        head = LinearHead(cfg["n_filters"], n_classes=cfg["n_classes"], tau=cfg["tau"])
    else:
        raise ValueError(f"unknown head: {cfg['head']}")
    return GaborNet(gf, head)


def save_model(model, path):
    """Portable checkpoint: architecture config + weights in one file.

    Loadable on any machine with only `torch` and this `models.py` -- no need to
    remember hyperparameters or match a pruned architecture by hand.
    """
    torch.save({"config": get_config(model), "state_dict": model.state_dict()}, path)


def load_model(path, map_location="cpu"):
    """Rebuild the exact architecture from a `save_model` checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_from_config(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model