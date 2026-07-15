#!/usr/bin/env python3

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import eggroll_es_worker_v38a as subject


class _TinyIdentityWorker:
    def _partition_identity_v4(self, partition, named_tensors, chunk_bytes):
        digest = hashlib.sha256()
        count = 0
        elements = 0
        for name, tensor in named_tensors:
            digest.update(name.encode("utf-8"))
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            count += 1
            elements += tensor.numel()
        return {
            "partition": partition,
            "sha256": digest.hexdigest(),
            "parameter_count": count,
            "total_elements": elements,
            "chunk_bytes": chunk_bytes,
        }


class WorkerV38ATest(unittest.TestCase):
    def test_update_surface_is_restored_and_snapshot_surface_added(self):
        cls = subject.EqualUnitUpdateWorkerExtensionV38A
        self.assertIs(
            cls.prepare_sharded_seed_update_v4,
            subject.worker_v11c.ResidentSignAuditWorkerExtensionV11C
            .prepare_sharded_seed_update_v4,
        )
        self.assertTrue(callable(cls.save_selected_snapshot_v38a))
        self.assertTrue(callable(cls.load_selected_snapshot_v38a))
        self.assertIn(
            "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9",
            subject.FROZEN_LAYER_PLANS_V38A,
        )

    def test_snapshot_readback_rehashes_serialized_tensor_bytes(self):
        worker = _TinyIdentityWorker()
        selected = [
            ("layer.a", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("layer.b", torch.tensor([3.0], dtype=torch.float32)),
        ]
        expected = worker._partition_identity_v4("selected", selected, 4096)
        metadata = {
            "schema": "test",
            "final_identity_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.safetensors"
            save_file(dict(selected), good, metadata=metadata)
            result = subject.readback_selected_snapshot_v38a(
                worker, good, selected, metadata, expected, chunk_bytes=4096,
            )
            self.assertTrue(result["verified"])
            self.assertEqual(result["selected_identity"], expected)

            changed = Path(directory) / "changed.safetensors"
            changed_tensors = dict(selected)
            changed_tensors["layer.b"] = torch.tensor([4.0])
            save_file(changed_tensors, changed, metadata=metadata)
            with self.assertRaisesRegex(RuntimeError, "differ from live"):
                subject.readback_selected_snapshot_v38a(
                    worker, changed, selected, metadata, expected,
                    chunk_bytes=4096,
                )


if __name__ == "__main__":
    unittest.main()
