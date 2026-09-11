from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import torch_pruning as tp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.notebook import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from models import save_model


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        pos_weight: torch.Tensor,
        device: torch.device,
        optimizer: torch.optim.Optimizer = AdamW,
        lr: float = 1e-2,
        lr_scheduler_kwargs: dict = {"eta_min": 0.0},
        seed: int | None = None,
        run_name: str | None = None,
        distill_alpha: float = 1.0,
        distill_temp: float = 2.0,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.optimizer_cls = optimizer
        self.base_lr = lr
        self.pruner = None
        self.regularize_on = False

        # knowledge distillation: loss = alpha*BCE(hard) + (1-alpha)*KD(soft teacher).
        # alpha=1.0 disables KD entirely, so non-distill runs are unchanged.
        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp

        # linear LR warmup (0 disables): ramp from warmup_start_factor*lr to lr over
        # warmup_epochs, then hand off to cosine. Applied only to the plain train() path.
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # differential LRs: calm the churning center freqs, base LR on the head.
        # No weight decay on center_freq (it's a physical param). Bandwidth is now static-Q (not learned).
        # A frozen (requires_grad=False) filterbank drops out of the optimizer entirely.
        gabor = self.model.feature_extractor
        self.gabor_trainable = gabor.center_freq.requires_grad
        param_groups = []
        if self.gabor_trainable:
            param_groups.append({"params": [gabor.center_freq], "lr": lr * 0.1, "weight_decay": 0.0})
        param_groups.append({"params": self.model.head.parameters(), "lr": lr})
        self.optimizer = optimizer(param_groups, lr=lr)
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_sched = None

        # explicit run_name overrides the default "<head>/<timestamp>" grouping
        if run_name is not None:
            run_stem = Path(run_name)
        else:
            head_name = self.model.head.__class__.__name__.lower().replace("head", "")
            run_id = datetime.now().strftime("%m%d-%H%M%S")
            run_stem = Path(head_name) / run_id
        log_dir = Path("runs") / run_stem
        self.checkpoint_path = Path("checkpoints") / run_stem.with_suffix(".pt")
        self.writer = SummaryWriter(log_dir=log_dir)

        self.lr = lr
        self.epoch = 0

        # training args (constant): captured now, paired with final metrics in log_model_info
        self.train_args = {
            "lr": lr,
            "optimizer": self.optimizer.__class__.__name__,
            "pos_weight": float(pos_weight.mean()),
            "gabor_trainable": self.gabor_trainable,
            "distill_alpha": distill_alpha,
            "distill_temp": distill_temp,
            "warmup_epochs": warmup_epochs,
            **{f"sched_{k}": v for k, v in lr_scheduler_kwargs.items()},
        }
        if seed is not None:
            self.train_args["seed"] = seed
        # final/best test metrics, filled in by eval()
        self.last_metrics = {}
        self.best_auc = float("-inf")

        # reference snapshot + running path length for the Gabor movement diagnostic
        self.cf0 = self._gabor_state()
        self.cf_prev = self.cf0.clone()
        self.cf_path = torch.zeros_like(self.cf0)   # cumulative distance travelled (Hz)

    def _build_pruner(self, pruning_ratio, reg):
        """Build a BN-scale pruner whose roots are restricted to CNN convolutions."""
        self.model.eval()
        gf = self.model.feature_extractor
        example = torch.randn(1, 1, gf.sr, device=self.device)
        importance = tp.importance.BNScaleImportance()
        return tp.pruner.BNScalePruner(
            self.model,
            example,
            importance=importance,
            reg=reg,
            global_pruning=True,
            pruning_ratio=pruning_ratio,
            ignored_layers=[gf, self.model.head.fc],
            root_module_types=[torch.nn.Conv2d],
            unwrapped_parameters=[(gf.center_freq, 0)],
        )

    def _rebuild_optimizer(self):
        """Rebuild optimizer groups around the live parameters created by pruning."""
        gabor = self.model.feature_extractor
        param_groups = []
        if self.gabor_trainable:
            param_groups.append({"params": [gabor.center_freq], "lr": self.base_lr * 0.1, "weight_decay": 0.0})
        param_groups.append({"params": self.model.head.parameters(), "lr": self.base_lr})
        self.optimizer = self.optimizer_cls(param_groups, lr=self.base_lr)
        self.lr_sched = None

    def _apply_pruning(self):
        """Physically remove the selected CNN channels and refresh dependent state."""
        if self.pruner is None:
            raise RuntimeError("Build a pruner before applying pruning")
        self.pruner.step()
        self.model.head.out_channels = self.model.head.fc.in_features
        self._rebuild_optimizer()

    def _model_config(self):
        """Read the live model's architecture config straight from the modules."""
        gf = self.model.feature_extractor
        cfg = {
            "model": self.model.__class__.__name__,
            "feature_extractor": gf.__class__.__name__,
            "n_filters": int(gf.center_freq.shape[0]),
            "kernel_size": int(gf.t.shape[-1]),
            "sample_rate": int(gf.sr),
            "stride": int(gf.stride),
            "Q": float(gf.Q),
            "head": self.model.head.__class__.__name__,
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "n_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        # head-specific fields, discovered by attribute so we don't import the head classes
        head = self.model.head
        if hasattr(head, "tau"):
            cfg["head_tau"] = float(head.tau)
        if hasattr(head, "out_channels"):
            cfg["head_channels"] = int(head.out_channels)
        elif hasattr(head, "conv"):
            cfg["head_channels"] = int(head.conv.out_channels)
        return cfg

    def log_model_info(self):
        """Pin config + results to this run: HParams table (filterable) + a readable text dump."""
        hparams = {**self.train_args, **self._model_config()}
        metrics = {
            "hparam/roc_auc_test_best": self.best_auc,
            "hparam/roc_auc_test_last": self.last_metrics.get("roc_auc", float("nan")),
            "hparam/loss_test_last": self.last_metrics.get("loss", float("nan")),
        }

        # readable dump in the TEXT tab
        md = "### training args\n" + "\n".join(
            f"- **{k}**: `{v}`" for k, v in self.train_args.items()
        )
        md += "\n\n### model config\n" + "\n".join(
            f"- **{k}**: `{v}`" for k, v in self._model_config().items()
        )
        md += "\n\n### results\n" + "\n".join(
            f"- **{k}**: `{v}`" for k, v in metrics.items()
        )
        self.writer.add_text("model_info", md, self.epoch)

        # HParams tab: (config -> results) row, run_name="." keeps it in THIS run
        self.writer.add_hparams(hparams, metrics, run_name=".")

    def _distill_loss(self, out, y, teacher_logits):
        """Blend hard BCE with temperature-scaled KD; returns (loss, bce_val, kd_val).

        KD uses soft targets sigmoid(teacher/T) against the student logits at the
        same temperature, scaled by T^2 to keep gradient magnitudes comparable to
        the hard term. Clips with no teacher target (NaN) drop out of the KD mean.
        """
        bce = self.criterion(out, y)
        if teacher_logits is None or self.distill_alpha >= 1.0:
            return bce, bce.item(), 0.0

        # Sanitize missing targets to 0 BEFORE the graph: masking NaNs only after
        # they enter the loss still poisons backward (0 * NaN = NaN gradient).
        valid = ~torch.isnan(teacher_logits)
        if not valid.any():
            return bce, bce.item(), 0.0
        safe_logits = torch.where(valid, teacher_logits, torch.zeros_like(teacher_logits))

        T = self.distill_temp
        soft_target = torch.sigmoid(safe_logits / T)
        kd_per = F.binary_cross_entropy_with_logits(out / T, soft_target, reduction="none")
        kd = (kd_per * valid).sum() / valid.sum() * (T ** 2)  # zero-weight missing clips
        loss = self.distill_alpha * bce + (1 - self.distill_alpha) * kd
        return loss, bce.item(), float(kd.item())

    def train_epoch(self):
        self.model.train()
        total_loss = total_bce = total_kd = 0.0
        all_scores, all_targets = [], []
        distilling = False
        for batch in tqdm(self.train_dataloader, leave=False):
            # DistillDataset batches carry a 4th element: per-clip teacher logits
            if len(batch) == 4:
                x, y, lengths, teacher_logits = batch
                teacher_logits = teacher_logits.to(self.device)
                distilling = True
            else:
                x, y, lengths = batch
                teacher_logits = None
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x, lengths)
            loss, bce_val, kd_val = self._distill_loss(out, y, teacher_logits)
            self.optimizer.zero_grad()
            loss.backward()
            if self.pruner is not None and self.regularize_on:
                self.pruner.regularize(self.model)
            self.optimizer.step()
            total_loss += loss.item()
            total_bce += bce_val
            total_kd += kd_val
            all_scores.append(out.detach().cpu())
            all_targets.append(y.detach().cpu())

        n_batches = len(self.train_dataloader)
        self.writer.add_scalar("Loss/train", total_loss / n_batches, self.epoch)
        if distilling:
            self.writer.add_scalar("Loss/train_bce", total_bce / n_batches, self.epoch)
            self.writer.add_scalar("Loss/train_kd", total_kd / n_batches, self.epoch)

        scores, targets = torch.cat(all_scores), torch.cat(all_targets)
        self.writer.add_scalar("ROC_AUC/train", roc_auc_score(targets.numpy(), scores.numpy()), self.epoch)

        if self.gabor_trainable:
            self.writer.add_scalar("LR/gabor", self.optimizer.param_groups[0]["lr"], self.epoch)
        self.writer.add_scalar("LR/head", self.optimizer.param_groups[-1]["lr"], self.epoch)
        self.lr_sched.step()

    @torch.no_grad()
    def eval(self, dataloader: DataLoader):
        self.model.eval()
        total_loss, all_scores, all_targets = 0.0, [], []

        # capture the head's pre-linear pooled features [B, n_filters] via a forward pre-hook
        feats_buf = []
        handle = self.model.head.fc.register_forward_pre_hook(
            lambda m, inp: feats_buf.append(inp[0].detach().cpu())
        )
        try:
            for x, y, lengths in tqdm(dataloader, leave=False):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x, lengths)
                total_loss += self.criterion(out, y).item()
                all_scores.append(out.cpu())
                all_targets.append(y.cpu())
        finally:
            handle.remove()

        scores = torch.cat(all_scores)
        targets = torch.cat(all_targets)
        loss = total_loss / len(dataloader)
        roc_auc = roc_auc_score(targets.numpy(), scores.numpy())  # AUC uses raw scores

        self.writer.add_scalar(f"Loss/test", loss, self.epoch)
        self.writer.add_scalar(f"ROC_AUC/test", roc_auc, self.epoch)

        self.last_metrics = {"loss": loss, "roc_auc": roc_auc}
        self.best_auc = max(self.best_auc, roc_auc)

        self.log_feature_diagnostics(torch.cat(feats_buf))

    def log_feature_diagnostics(self, feats):
        """Probe the head's input: per-band variance and inter-band correlation (redundancy)."""
        var = feats.var(dim=0)
        self.writer.add_histogram(f"Diag/band_variance", var, self.epoch)
        self.writer.add_scalar(f"Diag/band_variance_mean", var.mean().item(), self.epoch)
        self.writer.add_scalar(f"Diag/band_dead_frac", (var < 1e-4).float().mean().item(), self.epoch)

        corr = torch.corrcoef(feats.T)
        n = corr.shape[0]
        offdiag = corr[~torch.eye(n, dtype=torch.bool)]
        offdiag = offdiag[~offdiag.isnan()]
        self.writer.add_histogram(f"Diag/band_offdiag_corr", offdiag, self.epoch)
        self.writer.add_scalar(f"Diag/band_abs_corr_mean",
                               offdiag.abs().mean().item(), self.epoch)

    def log_slim_metrics(self):
        """Track head BN scales, which are the channel scores used for slimming."""
        scales = [
            module.weight.detach().abs().cpu()
            for module in self.model.head.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        ]
        if not scales:
            return
        gamma = torch.cat(scales)
        self.writer.add_histogram("Slim/gamma", gamma, self.epoch)
        self.writer.add_scalar(
            "Slim/gamma_zero_frac", (gamma < 1e-3).float().mean().item(), self.epoch
        )
        self.writer.add_scalar("Slim/head_channels", gamma.numel(), self.epoch)

    @torch.no_grad()
    def _gabor_state(self):
        """Decode the learnable center frequencies to Hz (bandwidth is static-Q, so redundant)."""
        gf = self.model.feature_extractor
        cf = torch.sigmoid(gf.center_freq) * (gf.sr / 2)   # center freq, Hz
        return cf.detach().cpu()

    def log_gabor_metrics(self):
        """Diagnose whether the Gabor filters are adapting or just thrashing."""
        if not self.gabor_trainable:
            return  # frozen filterbank never moves; these diagnostics are meaningless
        gf = self.model.feature_extractor
        # grads persist from the last train batch (not zeroed after step): an "are they moving?" probe
        cf_grad = gf.center_freq.grad

        cf = self._gabor_state()
        step = (cf - self.cf_prev).abs()                             # per-epoch velocity (Hz)
        self.cf_path += step                                        # cumulative distance travelled
        drift = (cf - self.cf0).abs()                               # net movement from init (Hz)
        directed = (drift / self.cf_path.clamp(min=1e-6)).mean()   # ~1 purposeful, ~0 thrashing
        scramble = (torch.argsort(cf) != torch.argsort(self.cf0)).float().mean()  # rank crossings

        self.writer.add_scalar("Gabor/cf_drift_mean_hz", drift.mean().item(), self.epoch)
        self.writer.add_scalar("Gabor/cf_velocity_mean_hz", step.mean().item(), self.epoch)
        self.writer.add_scalar("Gabor/cf_directedness", directed.item(), self.epoch)
        self.writer.add_scalar("Gabor/cf_rank_scramble", scramble.item(), self.epoch)
        if cf_grad is not None:
            self.writer.add_scalar("Gabor/cf_grad_norm", cf_grad.norm().item(), self.epoch)
        self.writer.add_histogram("Gabor/cf_hz", cf, self.epoch)
        self.cf_prev = cf

    def _run_phase(self, epochs, warmup_epochs=0):
        """Run one training phase: optional linear warmup, then a fresh cosine decay."""
        warmup_epochs = min(max(warmup_epochs, 0), max(epochs - 1, 0))
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs, **self.lr_scheduler_kwargs
        )
        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=self.warmup_start_factor,
                total_iters=warmup_epochs,
            )
            self.lr_sched = SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            self.lr_sched = cosine
        for _ in trange(epochs, colour="green"):
            self.train_epoch()
            self.eval(self.test_dataloader)
            self.log_gabor_metrics()
            self.log_slim_metrics()
            self.epoch += 1

    def train(self, epochs):
        # linear warmup (if set) then smooth cosine decay over the rest of the run;
        # both LR groups share this scheduler, preserving their differential base LRs.
        self.train_args["sched_T_max"] = epochs
        self._run_phase(epochs, warmup_epochs=self.warmup_epochs)
        self.log_model_info()
        self.save_model()

    def train_with_slimming(
        self,
        sparsify_epochs,
        finetune_epochs,
        pruning_ratio=0.3,
        reg=1e-5,
    ):
        """Sparsify BN scales, prune the CNN once, then fine-tune the smaller model."""
        self.train_args.update(
            {
                "slim_reg": reg,
                "slim_pruning_ratio": pruning_ratio,
                "slim_sparsify_epochs": sparsify_epochs,
                "slim_finetune_epochs": finetune_epochs,
            }
        )
        self.pruner = self._build_pruner(pruning_ratio, reg)
        self.regularize_on = True
        self._run_phase(sparsify_epochs)

        self._apply_pruning()
        self.regularize_on = False
        self.pruner = None
        self._run_phase(finetune_epochs)
        self.log_model_info()
        self.save_model()

    def save_model(self, path: str | Path | None = None):
        path = Path(path) if path is not None else self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_model(self.model, path, training_args=self.train_args)
        return path