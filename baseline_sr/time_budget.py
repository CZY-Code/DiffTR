"""Retain only reconstructions completed within predeclared wall-clock budgets."""
import time
import torch


class TimeBudget:
    def __init__(self, limits, device):
        self.limits = dict(limits)
        self.device = torch.device(device)
        self.snapshots = {}
        self.metadata = {}
        self.sync()
        self.started = time.perf_counter()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def elapsed(self):
        self.sync()
        return time.perf_counter() - self.started

    def expired(self):
        return self.elapsed() >= max(self.limits.values())

    def observe(self, reconstruction, **state):
        """Snapshot copying is charged to the clock; never accept a late iterate."""
        now = self.elapsed()
        eligible = [name for name, limit in self.limits.items() if now < limit]
        if eligible:
            rec = reconstruction.detach().clone()
            now = self.elapsed()
            for name in eligible:
                if now <= self.limits[name]:
                    self.snapshots[name] = dict(reconstruction=rec, seconds=now,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                        **state)
        return now >= max(self.limits.values())


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
