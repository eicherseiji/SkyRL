"""Selection policies for shedding in-flight trajectories."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from skyrl.train.generators.base import TrainingPhase


@dataclass(frozen=True)
class TrajectoryCancellationCandidate:
    """One cancellable trajectory attempt owned by a generator."""

    task_id: str
    group_id: str | None
    repetition_id: int | None
    started_at_s: float
    training_phase: TrainingPhase | None


@runtime_checkable
class TrajectoryCancellationPolicy(Protocol):
    """Select individual trajectory attempts for adaptive shedding."""

    def select_trajectories(
        self,
        candidates: Sequence[TrajectoryCancellationCandidate],
        shed_count: int,
    ) -> Sequence[str]: ...


class YoungestTrainingTrajectoryCancellationPolicy:
    """Cancel at most ``shed_count`` of the newest training trajectories.

    Evaluation trajectories are deliberately excluded. They still share sampling
    admission and therefore drain behind a lower limit, but adaptive training
    pressure does not repeatedly throw away evaluation results.
    """

    def select_trajectories(
        self,
        candidates: Sequence[TrajectoryCancellationCandidate],
        shed_count: int,
    ) -> Sequence[str]:
        if shed_count <= 0:
            return ()

        newest_first = sorted(
            (candidate for candidate in candidates if candidate.training_phase == "train"),
            key=lambda candidate: candidate.started_at_s,
            reverse=True,
        )
        return tuple(candidate.task_id for candidate in newest_first[:shed_count])
