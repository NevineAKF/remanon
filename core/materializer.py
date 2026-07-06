"""Band B — HBM3 tensor materialisation (stub)."""

from __future__ import annotations

from contracts.contract_a import HBM3Handle, Materializer


class CPUMaterializer(Materializer):
    """CPU-only stub; GPU path gated behind REMANON_GPU=1 (not yet implemented)."""

    def materialize(
        self,
        checkpoint_uri: str,
        handle: HBM3Handle,
        dtype: str = "bfloat16",
    ) -> None:
        # TODO: ROCm / hipBLAS loading path
        raise NotImplementedError(
            f"materialize({checkpoint_uri!r}, handle={handle!r}, dtype={dtype!r}) — stub"
        )

    def evict(self, handle: HBM3Handle) -> None:
        # TODO: release HBM3 region
        raise NotImplementedError(f"evict(handle={handle!r}) — stub")
