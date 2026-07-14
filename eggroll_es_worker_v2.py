"""Fail-closed worker methods for the corrected anchored EGGROLL recipe.

The frozen upstream worker restores a BF16 perturbation by adding the negative
noise.  BF16 addition is not reversible, so this extension deliberately makes
that path unavailable.  Each engine keeps a CPU reference of its committed
weights and restores from it with ``copy_`` after every perturbed rollout.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "es-at-scale"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from es_at_scale.utils.worker_extension import (  # noqa: E402
    WorkerExtension,
)


class ExactAuditWorkerExtension(WorkerExtension):
    """Use exact CPU references and expose byte-exact identity checks."""

    def restore_self_weights(self, seed, sigma):
        del seed, sigma
        raise RuntimeError(
            "subtractive perturbation restore is forbidden in anchored v2; "
            "use restore_self_weights_exact"
        )

    @staticmethod
    def _update_tensor_hash(digest, name, tensor, chunk_bytes):
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"cannot byte-hash noncontiguous parameter {name!r}"
            )
        metadata = json.dumps({
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8)
        byte_count = int(raw.numel())
        digest.update(byte_count.to_bytes(8, "little"))
        for start in range(0, byte_count, chunk_bytes):
            block = raw[start:start + chunk_bytes].cpu().numpy().tobytes()
            digest.update(block)
        return byte_count

    def save_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        """Replace the CPU reference with the engine's committed weights."""
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("digest chunk size must be positive")
        # Release the old reference before refreshing so a committed update
        # never transiently requires two complete host copies per engine.
        self.exact_reference_weights = {}
        digest = hashlib.sha256()
        parameter_count = 0
        total_bytes = 0
        for name, parameter in self.model_runner.model.named_parameters():
            reference = parameter.detach().to(device="cpu", copy=True)
            self.exact_reference_weights[name] = reference
            total_bytes += self._update_tensor_hash(
                digest, name, reference, chunk_bytes,
            )
            parameter_count += 1
        self.exact_reference_identity = {
            "schema": "eggroll-es-weight-state-sha256-v2",
            "sha256": digest.hexdigest(),
            "parameter_count": parameter_count,
            "total_bytes": total_bytes,
        }
        return dict(self.exact_reference_identity)

    def restore_self_weights_exact(self):
        """Restore every parameter exactly from this engine's CPU reference."""
        references = getattr(self, "exact_reference_weights", None)
        if not references:
            raise RuntimeError("exact weight reference has not been captured")
        seen = set()
        with torch.no_grad():
            for name, parameter in self.model_runner.model.named_parameters():
                if name not in references:
                    raise RuntimeError(
                        f"exact reference is missing parameter {name!r}"
                    )
                reference = references[name]
                if reference.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"exact reference dtype changed for {name!r}"
                    )
                if tuple(reference.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"exact reference shape changed for {name!r}"
                    )
                parameter.data.copy_(reference, non_blocking=False)
                seen.add(name)
        extra = set(references) - seen
        if extra:
            raise RuntimeError(
                f"exact reference contains unknown parameters: {sorted(extra)}"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def verify_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        """Byte-hash current weights and require equality with the reference."""
        references = getattr(self, "exact_reference_weights", None)
        reference_identity = getattr(self, "exact_reference_identity", None)
        if not references or not reference_identity:
            raise RuntimeError("exact weight reference has not been captured")
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("digest chunk size must be positive")
        digest = hashlib.sha256()
        parameter_count = 0
        total_bytes = 0
        seen = set()
        for name, parameter in self.model_runner.model.named_parameters():
            if name not in references:
                raise RuntimeError(
                    f"exact reference is missing parameter {name!r}"
                )
            current = parameter.detach().cpu()
            if not torch.equal(current, references[name]):
                raise RuntimeError(
                    f"current weights differ from exact reference at {name!r}"
                )
            total_bytes += self._update_tensor_hash(
                digest, name, current, chunk_bytes,
            )
            parameter_count += 1
            seen.add(name)
        if set(references) != seen:
            raise RuntimeError("exact reference parameter names changed")
        current_identity = {
            "schema": "eggroll-es-weight-state-sha256-v2",
            "sha256": digest.hexdigest(),
            "parameter_count": parameter_count,
            "total_bytes": total_bytes,
        }
        if current_identity != reference_identity:
            raise RuntimeError("current weight digest differs from reference")
        return {
            "schema": "eggroll-es-exact-reference-check-v2",
            "passed": True,
            "reference": dict(reference_identity),
            "current": current_identity,
        }

    def weight_state_sha256(self, chunk_bytes=64 * 1024 * 1024):
        """Hash all local parameter bytes without retaining a host checkpoint.

        Each tensor is copied to the CPU in bounded chunks.  Metadata is part
        of the digest, so name, dtype, shape, or byte changes are detected.
        This is intentionally expensive: it is the mandatory alpha-zero audit,
        not a per-generation telemetry path.
        """
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("digest chunk size must be positive")
        digest = hashlib.sha256()
        parameter_count = 0
        total_bytes = 0
        for name, parameter in self.model_runner.model.named_parameters():
            byte_count = self._update_tensor_hash(
                digest, name, parameter.detach(), chunk_bytes,
            )
            parameter_count += 1
            total_bytes += byte_count
        return {
            "schema": "eggroll-es-weight-state-sha256-v2",
            "sha256": digest.hexdigest(),
            "parameter_count": parameter_count,
            "total_bytes": total_bytes,
        }
