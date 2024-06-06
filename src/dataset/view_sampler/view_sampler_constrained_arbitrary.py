from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .three_view_hack import add_third_context_index
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerConstrainedArbitraryCfg:
    name: Literal["constrained_arbitrary"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | None
    target_views: list[int] | None
    context_target_max_distance: int


class ViewSamplerConstrainedArbitrary(ViewSampler[ViewSamplerConstrainedArbitraryCfg]):
    """
    A ViewSampler that is more flexible than the Bounded, but less flexible than the Arbitrary sampler.
    It samples target view that are either between the first and last context view, or at most 
    `context_target_max_distance` before or after.
    """
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape

        index_context = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_context_views,),
            device=device,
        )

        # Allow the context views to be fixed.
        if self.cfg.context_views is not None:
            index_context = torch.tensor(
                self.cfg.context_views, dtype=torch.int64, device=device
            )

            if self.cfg.num_context_views == 3 and len(self.cfg.context_views) == 2:
                index_context = add_third_context_index(index_context)
            else:
                assert len(self.cfg.context_views) == self.cfg.num_context_views

        first_context_idx = torch.min(index_context)
        last_context_idx = torch.max(index_context)
        index_target = torch.randint(
            max(0, first_context_idx - self.cfg.context_target_max_distance),
            min(num_views, last_context_idx + self.cfg.context_target_max_distance),
            size=(self.cfg.num_target_views,),
            device=device,
        )

        # Allow the target views to be fixed.
        if self.cfg.target_views is not None:
            assert len(self.cfg.target_views) == self.cfg.num_target_views
            index_target = torch.tensor(
                self.cfg.target_views, dtype=torch.int64, device=device
            )

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
