"""Selection policies for shedding in-flight trajectory groups."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from skyrl.train.generators.base import TrainingPhase


@dataclass(frozen=True)
class TrajectoryGroupCancellationCandidate:
    """A cancellable prompt group owned by a generator.

    ``active_trajectory_count`` is the number of live repetitions in the
    group. A selection always cancels the whole group, even when doing so
    exceeds the requested trajectory count.
    """

    group_id: str
    started_at_s: float
    active_trajectory_count: int
    training_phase: TrainingPhase | None


@runtime_checkable
class TrajectoryGroupCancellationPolicy(Protocol):
    """Select complete prompt groups for a requested amount of shedding."""

    def select_groups(
        self,
        candidates: Sequence[TrajectoryGroupCancellationCandidate],
        shed_count: int,
    ) -> Sequence[str]: ...


class YoungestTrainingGroupCancellationPolicy:
    """Cancel the newest training groups until ``shed_count`` is covered.

    Evaluation groups are deliberately excluded. They still share sampling
    admission and therefore drain behind a lower limit, but adaptive training
    pressure does not repeatedly throw away evaluation results.
    """

    def select_groups(
        self,
        candidates: Sequence[TrajectoryGroupCancellationCandidate],
        shed_count: int,
    ) -> Sequence[str]:
        if shed_count <= 0:
            return ()

        selected: list[str] = []
        selected_trajectories = 0
        for candidate in sorted(
            (candidate for candidate in candidates if candidate.training_phase == "train"),
            key=lambda candidate: candidate.started_at_s,
            reverse=True,
        ):
            selected.append(candidate.group_id)
            selected_trajectories += candidate.active_trajectory_count
            if selected_trajectories >= shed_count:
                break
        return tuple(selected)
