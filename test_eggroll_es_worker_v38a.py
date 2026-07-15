#!/usr/bin/env python3

import unittest

import eggroll_es_worker_v38a as subject


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


if __name__ == "__main__":
    unittest.main()
