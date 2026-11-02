"""Device-side row scheduling for shared-policy physical scene admission."""

import torch


class AdmissionSchedule:
    def __init__(
        self, count, device, *, attempts=16, warmup_steps=300, cadence=32, final_warmup_steps=None
    ):
        final_warmup_steps = warmup_steps if final_warmup_steps is None else final_warmup_steps
        if min(count, attempts, warmup_steps, cadence, final_warmup_steps) < 1:
            raise ValueError("Admission schedule dimensions and budgets must be positive")
        self.attempts, self.warmup_steps, self.cadence = attempts, warmup_steps, cadence
        self.final_warmup_steps = final_warmup_steps
        self.max_physics_steps = (attempts - 1) * warmup_steps + final_warmup_steps
        self.tick = 0
        self.attempt_index = torch.zeros(count, device=device, dtype=torch.int64)
        self.age = torch.zeros_like(self.attempt_index)
        self.limit = torch.full_like(
            self.age, final_warmup_steps if attempts == 1 else warmup_steps
        )
        self.waiting = torch.ones(count, device=device, dtype=torch.bool)
        self.admitted = torch.zeros_like(self.waiting)
        self.needs_layout = torch.zeros_like(self.waiting)
        self.exhausted = torch.zeros_like(self.waiting)

    @property
    def active(self):
        return self.waiting | self.needs_layout

    def advance(self, steps=1):
        if steps < 1:
            raise ValueError("Physical admission steps must advance")
        if self.tick + steps > self.max_physics_steps:
            raise ValueError("Shared physical admission budget exceeded")
        self.age += self.waiting.long() * steps
        self.tick += steps
        torch._assert_async(
            ((self.age <= self.limit) | ~self.waiting).all(),
            "Admission driver exceeded a row's physical warmup budget",
        )

    def next_wait(self):
        remaining = (self.limit - self.age).masked_fill(~self.waiting, self.max_physics_steps)
        return torch.minimum(
            remaining.amin(),
            remaining.new_tensor(
                min(self.cadence - self.tick % self.cadence, self.max_physics_steps - self.tick)
            ),
        )

    def validation_rows(self, ready):
        if ready.shape != self.waiting.shape or ready.dtype != torch.bool:
            raise ValueError("Admission readiness must contain one boolean per source row")
        due = (ready & (self.tick % self.cadence == 0)) | (self.age >= self.limit)
        return torch.where(self.waiting & due)[0]

    def finish(self, rows, accepted):
        if accepted.shape != rows.shape or accepted.dtype != torch.bool:
            raise ValueError("Admission decisions must match the validated GPU rows")
        self.waiting[rows] = False
        self.admitted[rows] = accepted
        retry = (
            ~accepted
            & (self.attempt_index[rows] + 1 < self.attempts)
            & (self.tick < self.max_physics_steps)
        )
        self.needs_layout[rows] = retry
        self.exhausted[rows] = ~accepted & ~retry

    def layout_rows(self):
        return torch.where(self.needs_layout)[0]

    def proposals(self, rows, available):
        if available.shape != rows.shape or available.dtype != torch.bool:
            raise ValueError("Layout availability must match the requested GPU rows")
        if self.tick >= self.max_physics_steps:
            raise ValueError("Shared physical admission budget exceeded")
        torch._assert_async(
            self.needs_layout[rows].all(), "Layout reset requested for an ineligible row"
        )
        self.attempt_index[rows] += 1
        torch._assert_async(
            (self.attempt_index[rows] < self.attempts).all(), "Admission retry budget exceeded"
        )
        reset = rows[available]
        self.limit[rows] = torch.where(
            self.attempt_index[rows] + 1 == self.attempts,
            self.final_warmup_steps,
            self.warmup_steps,
        )
        self.waiting[reset] = True
        self.age[reset] = 0
        self.needs_layout[rows] = ~available & (self.attempt_index[rows] + 1 < self.attempts)
        self.exhausted[rows] = ~available & ~self.needs_layout[rows]
        return reset

    def expire_shared_budget(self):
        if self.tick < self.max_physics_steps:
            raise ValueError("Shared physical admission budget has not expired")
        self.exhausted |= self.active
        self.waiting.zero_()
        self.needs_layout.zero_()
