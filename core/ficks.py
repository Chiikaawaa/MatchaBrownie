import numpy as np

class FicksSolver:
   
    def __init__(self, n_voxels: int, box_size: float, D: float, dt: float):
        self.n_voxels = n_voxels
        self.box_size = box_size
        self.D        = D
        self.dt       = dt
        self._dx      = box_size / n_voxels
        self.C        = np.zeros((n_voxels, n_voxels, n_voxels))
        self.history  = []

        self.r = D * dt / (self._dx ** 2)
        if self.r >= 1/6:
            raise ValueError(
                f"Unstable: r={self.r:.6f} must be < 0.1667. "
                f"Reduce dt or increase n_voxels."
            )

    def set_initial_condition(self, positions: np.ndarray):
       
        self.C  = np.zeros((self.n_voxels, self.n_voxels, self.n_voxels))
        half    = self.box_size / 2
        indices = np.floor(
            (positions + half) / self._dx
        ).astype(int)
        indices = np.clip(indices, 0, self.n_voxels - 1)
        for idx in indices:
            self.C[idx[0], idx[1], idx[2]] += 1
        self.C = self.C / (self._dx ** 3)

    def step(self):
      
        C  = self.C
        dx = self._dx

        C_pad = np.pad(C, pad_width=1, mode='edge')

        laplacian = (
            C_pad[2:,  1:-1, 1:-1] + C_pad[:-2, 1:-1, 1:-1] +
            C_pad[1:-1, 2:,  1:-1] + C_pad[1:-1, :-2, 1:-1] +
            C_pad[1:-1, 1:-1,  2:] + C_pad[1:-1, 1:-1, :-2] -
            6 * C
        ) / dx**2

        self.C = C + self.D * self.dt * laplacian

    def run(self, n_steps: int, snapshot_every: int = 10):
        for i in range(n_steps):
            self.step()
            if i % snapshot_every == 0:
                self.history.append(self.C.copy())

    def get_slice(self, axis: int = 2, index: int = None) -> np.ndarray:
        if index is None:
            index = self.n_voxels // 2
        if axis == 0:
            return self.C[index, :, :]
        elif axis == 1:
            return self.C[:, index, :]
        else:
            return self.C[:, :, index]

    def get_total_concentration(self) -> float:
        return float(self.C.sum() * self._dx**3)

    def reset(self):
        self.C       = np.zeros((self.n_voxels, self.n_voxels, self.n_voxels))
        self.history = []