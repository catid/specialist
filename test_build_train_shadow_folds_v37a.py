#!/usr/bin/env python3
import unittest

import build_train_shadow_folds_v37a as folds
import run_sft_train_only_control_v36a as runtime


class TrainShadowFoldsV37ATest(unittest.TestCase):
    def test_complete_conflict_disjoint_crossfit(self):
        value = folds.build()
        self.assertEqual(len(value["folds"]), 5)
        self.assertEqual(value["policy"]["conflict_units"], 259)
        self.assertFalse(
            value["policy"]["external_evaluation_ood_holdout_or_benchmark_opened"]
        )
        self.assertEqual(value["coverage"]["all_fold_shadow_dev_rows"], 531)
        self.assertEqual(value["coverage"]["all_fold_train_rows"], 531 * 4)
        self.assertEqual(
            [
                (item["train"]["rows"], item["shadow_dev"]["rows"])
                for item in value["folds"]
            ],
            [(372, 159), (408, 123), (452, 79), (448, 83), (444, 87)],
        )
        self.assertTrue(all(
            item["train_dev_conflict_unit_intersection"] == 0
            and item["train_dev_document_sha256_intersection"] == 0
            for item in value["folds"]
        ))
        self.assertEqual(
            value["content_sha256_before_self_field"],
            runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
