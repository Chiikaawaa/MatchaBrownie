import numpy as np


class ConcentrationField:
    

    def __init__(self, box_size: float, n_voxels: int = 20):
        self.box_size   = box_size
        self.n_voxels   = n_voxels
        self.voxel_size = box_size / n_voxels
        self.grid       = np.zeros((n_voxels, n_voxels, n_voxels))
        self.history    = []

    def update(self, positions: np.ndarray, active_mask: np.ndarray = None):
        
        
        self.grid = np.zeros((self.n_voxels, self.n_voxels, self.n_voxels))
        half = self.box_size / 2

        pts = positions if active_mask is None else positions[active_mask]
        if len(pts) == 0:
            return

        indices = np.floor(
            (pts + half) / self.voxel_size
        ).astype(int)
        indices = np.clip(indices, 0, self.n_voxels - 1)

        n = self.n_voxels
        flat_idx = (indices[:, 0] * n**2 + 
                   indices[:, 1] * n + 
                   indices[:, 2])

        self.grid = np.bincount(flat_idx, minlength=n**3).reshape(n, n, n).astype(float)

    def snapshot(self):
        self.history.append(self.grid.copy())

    def get_slice(self, axis: int = 2, index: int = None) -> np.ndarray:
        
        if index is None:
            index = self.n_voxels // 2
        if axis == 0:
            return self.grid[index, :, :]
        elif axis == 1:
            return self.grid[:, index, :]
        else:
            return self.grid[:, :, index]

    def get_flux_across_membrane(self, membrane_axis: int = 2):
        
        mid = self.n_voxels // 2
        if membrane_axis == 2:
            left  = self.grid[:, :, :mid].sum()
            right = self.grid[:, :, mid:].sum()
        elif membrane_axis == 1:
            left  = self.grid[:, :mid, :].sum()
            right = self.grid[:, mid:, :].sum()
        else:
            left  = self.grid[:mid, :, :].sum()
            right = self.grid[mid:, :, :].sum()
        return int(left), int(right)


class FirstPassageTracker:
    def __init__(self, axis: int, threshold: float, dt: float, n_particles: int):
        self.axis              = axis
        self.threshold         = threshold
        self.dt                = dt
        self.n_particles_total = n_particles
        self.fpt               = {}            
        self._already_crossed: set = set()

    def update(self, positions: np.ndarray, step: int, active_mask: np.ndarray = None):
        candidate_idx = np.arange(len(positions))

        if step == 0:
            self._already_crossed = set(
                candidate_idx[positions[candidate_idx, self.axis] >= self.threshold]
            )
            return

        crossed_mask = positions[candidate_idx, self.axis] >= self.threshold
        for idx in candidate_idx[crossed_mask]:
            if idx not in self.fpt and idx not in self._already_crossed:
                self.fpt[idx] = step

    def get_times(self) -> np.ndarray:
        return np.array(list(self.fpt.values())) * self.dt

    def get_stats(self) -> dict:
        times = self.get_times()
        if len(times) == 0:
            return {"n_arrived": 0}

        mean   = np.mean(times)
        median = np.median(times)
        std    = np.std(times, ddof=1)
        n      = len(times)

        ci_margin = 1.96 * (std / np.sqrt(n)) if n > 0 else 0

        return {
            "n_arrived": n,
            "mean":      mean,
            "median":    median,
            "std":       std,
            "ci_lower":  mean - ci_margin,
            "ci_upper":  mean + ci_margin,
            "t_10pct":   np.percentile(times, 10),
            "t_50pct":   np.percentile(times, 50),
            "t_90pct":   np.percentile(times, 90),
            "t_95pct":   np.percentile(times, 95),
        }