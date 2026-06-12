import numpy as np

class Simulator:
    def __init__(
        self,
        n_particles:         int,
        D:                   float,
        dt:                  float,
        geometry             = None,
        membrane             = None,
        concentration_field  = None,
        fpt_tracker          = None,
        transporters         = None,
    ):
        self.n_particles         = n_particles
        self.D                   = D
        self.dt                  = dt
        self.geometry            = geometry
        self.membrane            = membrane
        self.concentration_field = concentration_field
        self.fpt_tracker         = fpt_tracker
        self.transporters        = transporters or []
        self.positions           = np.zeros((n_particles, 3))
        self.history             = [self.positions.copy()]
        self._active_history     = [np.ones(n_particles, dtype=bool)]
        self.time                = 0.0
        self.step_count          = 0

    def step(self):
        if self.geometry is not None and self.geometry._active is None:
            active_mask = np.ones(self.n_particles, dtype=bool)
        else:
            active_mask = (
                self.geometry.active_mask
                if self.geometry is not None
                else np.ones(self.n_particles, dtype=bool)
            )

        n_active = np.sum(active_mask)
        if n_active == 0:
            self.history.append(self.positions.copy())
            self._active_history.append(active_mask.copy())
            self.time += self.dt
            self.step_count += 1
            return

        sigma = np.sqrt(2 * self.D * self.dt)
        noise = np.zeros((self.n_particles, 3))
        noise[active_mask] = np.random.normal(0, sigma, size=(n_active, 3))

        total_drift = np.zeros((self.n_particles, 3))
        if self.transporters:
            box_size = (
                float(self.geometry.bounds[self.transporters[0].params.axis, 1] -
                      self.geometry.bounds[self.transporters[0].params.axis, 0])
                if self.geometry is not None else 1e-5
            )
            for transporter in self.transporters:
                drift = transporter.compute_drift(
                    self.positions, self.dt, box_size, self.concentration_field
                )
                total_drift[active_mask] += drift[active_mask]

        old_positions = self.positions.copy()
        potential_positions = old_positions.copy()
        potential_positions[active_mask] += noise[active_mask] + total_drift[active_mask]

        if self.membrane is not None:
            self.positions = self.membrane.apply(old_positions, potential_positions)
        else:
            self.positions = potential_positions

        if self.geometry is not None:
            self.positions = self.geometry.apply(old_positions, self.positions)

        active_mask = (
            self.geometry.active_mask
            if self.geometry is not None
            else np.ones(self.n_particles, dtype=bool)
        )

        if self.concentration_field is not None:
            self.concentration_field.update(self.positions, active_mask)
            if self.step_count % 10 == 0:
                self.concentration_field.snapshot()

        if self.fpt_tracker is not None:
            self.fpt_tracker.update(self.positions, self.step_count, active_mask)
 
        self.history.append(self.positions.copy())
        self._active_history.append(active_mask.copy())
        self.time += self.dt
        self.step_count += 1

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def run_ensemble(self, n_steps: int, n_runs: int, seed_position: float) -> dict:
        all_msd = []
        sigma = np.sqrt(2 * self.D * self.dt)

        for _ in range(n_runs):
            self.reset()
            if seed_position is not None:
                self.positions[:,2] = seed_position 
            if self.geometry is not None:
                self.positions = self.geometry.apply(np.zeros_like(self.positions), self.positions)
    
            origin = self.positions.copy()
            msd = np.zeros(n_steps + 1)
            msd[0] = 0.0
    
            all_noise = np.random.normal(0, sigma, size=(n_steps, self.n_particles, 3))
            origin = self.positions.copy()
            msd = np.zeros(n_steps + 1)
            msd[0] = 0.0

            all_noise = np.random.normal(0, sigma, size=(n_steps, self.n_particles, 3))

            for i in range(n_steps):
                active_mask = (
                    self.geometry.active_mask
                    if self.geometry is not None
                    else np.ones(self.n_particles, dtype=bool)
                )

                n_active = np.sum(active_mask)
                if n_active == 0:
                    self.time += self.dt
                    self.step_count += 1
                    continue

                noise = np.zeros((self.n_particles, 3))
                noise[active_mask] = all_noise[i][active_mask]

                total_drift = np.zeros((self.n_particles, 3))
                if self.transporters:
                    box_size = (
                        float(self.geometry.bounds[self.transporters[0].params.axis, 1] -
                              self.geometry.bounds[self.transporters[0].params.axis, 0])
                        if self.geometry is not None else 1e-5
                    )
                    for transporter in self.transporters:
                        drift = transporter.compute_drift(
                            self.positions, self.dt, box_size, self.concentration_field
                        )
                        total_drift[active_mask] += drift[active_mask]

                old_positions = self.positions.copy()
                potential_positions = old_positions.copy()
                potential_positions[active_mask] += noise[active_mask] + total_drift[active_mask]

                if self.membrane is not None:
                    self.positions = self.membrane.apply(old_positions, potential_positions)
                else:
                    self.positions = potential_positions

                if self.geometry is not None:
                    self.positions = self.geometry.apply(old_positions, self.positions)

                active_mask = (
                    self.geometry.active_mask
                    if self.geometry is not None
                    else np.ones(self.n_particles, dtype=bool)
                )

                if self.concentration_field is not None:
                    self.concentration_field.update(self.positions, active_mask)
                    if self.step_count % 10 == 0:
                        self.concentration_field.snapshot()

                if self.fpt_tracker is not None:
                    self.fpt_tracker.update(self.positions, self.step_count, active_mask)

                disp = self.positions[active_mask] - origin[active_mask]
                msd[i+1] = np.mean(np.sum(disp**2, axis=1)) if active_mask.any() else np.nan

                self.time += self.dt
                self.step_count += 1

            all_msd.append(msd)
            self.history = [self.positions.copy()]
            self._active_history = [active_mask.copy()]

        all_msd = np.array(all_msd)
        return {
            "mean": np.nanmean(all_msd, axis=0),
            "std":  np.nanstd(all_msd,  axis=0),
            "time": np.arange(n_steps + 1) * self.dt,
        }

    def get_msd(self) -> np.ndarray:
        history_arr = np.array(self.history)      
        active_arr  = np.array(self._active_history)  
        origin      = history_arr[0]
        disp_sq     = np.sum((history_arr - origin)**2, axis=2) 
    
        # masked mean — where active=False, set to nan then nanmean
        disp_sq_masked = np.where(active_arr, disp_sq, np.nan)
        return np.nanmean(disp_sq_masked, axis=1)

    def reset(self):
        self.positions       = np.zeros((self.n_particles, 3))
        self.history         = [self.positions.copy()]
        self._active_history = [np.ones(self.n_particles, dtype=bool)]
        self.time            = 0.0
        self.step_count      = 0

        if self.membrane is not None:
            self.membrane.n_attempts  = 0
            self.membrane.n_crossings = 0

        if self.geometry is not None:
            self.geometry.reset_counters()

        if self.fpt_tracker is not None:
            self.fpt_tracker.fpt              = {}
            self.fpt_tracker._already_crossed = set()

        if self.concentration_field is not None:
            self.concentration_field.grid = np.zeros((
                self.concentration_field.n_voxels,
                self.concentration_field.n_voxels,
                self.concentration_field.n_voxels,
            ))
            self.concentration_field.history = []

        for transporter in self.transporters:
            transporter.reset()

    def seed_left_half(self, half: float):
        self.positions[:, 2] = np.random.uniform(-half, -half / 1.5, self.n_particles)
        self.history = [self.positions.copy()]
        self._active_history = [np.ones(self.n_particles, dtype=bool)]
        self.time = 0.0
        self.step_count = 0

    def seed_at_z(self, z_value: float):
        self.positions[:, 2] = z_value
        self.history = [self.positions.copy()]
        self._active_history = [np.ones(self.n_particles, dtype=bool)]
        self.time = 0.0
        self.step_count = 0