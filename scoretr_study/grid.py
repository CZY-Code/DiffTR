"""Memory-bounded exact ScoreTR grid search (same candidate grid and energy)."""
import torch
from EBM.energy4Imgd import ScoreMatching


class BoundedScoreMatching(ScoreMatching):
    @torch.no_grad()
    def search(self, idx, grid):
        # Coordinate features are invariant over intensity candidates.
        t = idx.float() / self.norm_factors
        z = torch.cat([net(self.encoding(t[:, d:d+1])) for d, net in enumerate(
            [self.U_Net_mu, self.V_Net_mu, self.W_Net_mu])], -1)
        z = self.layers['z_enc'](z)
        best = torch.full((len(idx),), -torch.inf, device=idx.device)
        chosen = torch.zeros_like(best)
        for first in range(0, len(grid), 8):
            x = grid[first:first+8]
            xe = self.layers['x_enc'](x.unsqueeze(-1))
            ze = z.unsqueeze(0).expand(len(x), -1, -1)
            energy = -self.layers['output'](torch.cat([ze, xe], -1)).pow(2).sum(-1)
            value, pos = energy.max(0)
            update = value > best
            chosen = torch.where(update, x.gather(0, pos[None]).squeeze(0), chosen)
            best = torch.maximum(best, value)
        return chosen, best

    def grid_search_posterior4train(self, idx, x, epsilon, delta=.329):
        grid = torch.arange(-delta, delta + epsilon/2, epsilon,
                            device=self.device, dtype=torch.float32)[:, None] + x.detach()[None]
        # Original loss never uses gradients of the returned log probability;
        # argmax-selected candidates have no parameter gradient either.
        return self.search(idx, grid)

    def grid_search_posterior(self, idx, x_range, epsilon):
        grid = torch.arange(x_range[0], x_range[1] + epsilon/2, epsilon,
                            device=self.device, dtype=torch.float32)[:, None].expand(-1, len(idx))
        return self.search(idx, grid)[0]
