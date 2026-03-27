"""Tracking and storing motion correction optimization state."""

import torch


class OptimizationState:
    """Dataclass storing optimization state at a single iteration.

    Parameters
    ----------
    deformation_field : torch.Tensor
        The deformation field at this checkpoint with shape (2, nt, nh, nw) where
        2 corresponds to (y, x) shifts.
    loss : float
        The loss value at this checkpoint.
    step : int
        The optimization step number at this checkpoint.

    Methods
    -------
    as_dict() -> dict
        Returns a dictionary representation of the checkpoint.
    """

    deformation_field: torch.Tensor  # (yx, nt, nh, nw)
    loss: float
    step: int

    def __init__(self, deformation_field: torch.Tensor, loss: float, step: int):
        self.deformation_field = deformation_field.cpu()
        self.loss = loss
        self.step = step

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation of the optimization checkpoint.

        Returns
        -------
        dict
            A dictionary representation of the optimization checkpoint.
        """
        return {
            "deformation_field": self.deformation_field.tolist(),
            "loss": self.loss,
            "step": self.step,
        }


class OptimizationTracker:
    """Tracking and storing motion correction optimization state.

    Parameters
    ----------
    optimization_checkpoints : list[OptimizationState]
        List of optimization checkpoints recorded during the optimization process.
    sample_every_n_steps : int
        Frequency of sampling checkpoints during optimization.
    total_steps : int
        Total number of optimization steps.

    Methods
    -------
    sample_this_step(step: int) -> bool
        Determines if a checkpoint should be sampled at the given step.
    add_checkpoint(deformation_field: torch.Tensor, loss: float, step: int) -> None
        Adds a new optimization checkpoint.
    as_dict() -> dict
        Returns a dictionary representation of the optimization trajectory.
    to_json(filepath: str) -> None
        Saves the optimization trajectory to a JSON file.
    """

    checkpoints: list[OptimizationState]
    sample_every_n_steps: int
    total_steps: int

    def __init__(self, sample_every_n_steps: int, total_steps: int):
        self.checkpoints = []
        self.sample_every_n_steps = sample_every_n_steps
        self.total_steps = total_steps

    def sample_this_step(self, step: int) -> bool:
        """
        Determines if a checkpoint should be sampled at the given step.

        Parameters
        ----------
        step: int
            The optimization step number.

        Returns
        -------
        bool
            True if a checkpoint should be sampled at the given step, False otherwise.
        """
        return step % self.sample_every_n_steps == 0 or step == self.total_steps - 1

    def add_checkpoint(
        self, deformation_field: torch.Tensor, loss: float, step: int
    ) -> None:
        """
        Adds a new optimization checkpoint.

        Parameters
        ----------
        deformation_field: torch.Tensor
            The deformation field at this checkpoint with shape (2, nt, nh, nw) where
            2 corresponds to (y, x) shifts.
        loss: float
            The loss value at this checkpoint.
        step: int
            The optimization step number at this checkpoint.
        """
        self.checkpoints.append(OptimizationState(deformation_field, loss, step))

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation of the optimization trajectory.

        Returns
        -------
        dict
            A dictionary representation of the optimization trajectory.
        """
        return {
            "optimization_checkpoints": [cp.as_dict() for cp in self.checkpoints],
            "sample_every_n_steps": self.sample_every_n_steps,
            "total_steps": self.total_steps,
        }

    def to_json(self, filepath: str) -> None:
        """
        Saves the optimization trajectory to a JSON file.

        Parameters
        ----------
        filepath: str
            The path to the JSON file to save the optimization trajectory to.
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.as_dict(), f)
