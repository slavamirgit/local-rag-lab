import unittest
from unittest.mock import patch

from src.rag import fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_one_ranking_preserves_order_and_zero_id(self):
        self.assertEqual(fusion.reciprocal_rank_fusion([[7, 0, 10]]), [7, 0, 10])

    def test_two_rankings_combine_scores(self):
        # 2: 1/62 + 1/61; 7: 1/61 + 1/63; 5: 1/62; 10: 1/63.
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[7, 2, 10], [2, 5, 7]]), [2, 7, 5, 10]
        )

    def test_shared_candidate_outranks_each_first_result(self):
        self.assertEqual(fusion.reciprocal_rank_fusion([[9, 1], [8, 1]]), [1, 8, 9])

    def test_one_based_ranks(self):
        # ID 1: 1/3 + 0.4/2 > ID 9: 1/2. Zero-based ranks reverse this.
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[9, 1], [1]], k=1, weights=[1, 0.4]), [1, 9]
        )

    def test_default_k_is_60_and_uses_configuration(self):
        rankings = [[9, 5, 1], [7, 6, 1]]
        self.assertEqual(fusion.RRF_K, 60)
        self.assertEqual(fusion.reciprocal_rank_fusion(rankings), [1, 7, 9, 5, 6])
        self.assertEqual(
            fusion.reciprocal_rank_fusion(rankings),
            fusion.reciprocal_rank_fusion(rankings, k=60),
        )
        with patch.object(fusion, "RRF_K", 0.1):
            self.assertEqual(fusion.reciprocal_rank_fusion(rankings), [7, 9, 1, 5, 6])

    def test_custom_k(self):
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[9, 5, 1], [7, 6, 1]], k=0.1), [7, 9, 1, 5, 6]
        )

    def test_optional_weights(self):
        rankings = [[7, 2], [2, 5]]
        self.assertEqual(fusion.reciprocal_rank_fusion(rankings, weights=[1, 1]), [2, 7, 5])
        self.assertEqual(fusion.reciprocal_rank_fusion(rankings, weights=[1, 2]), [2, 5, 7])

    def test_zero_weight_adds_no_candidates(self):
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[7, 2], [2, 5]], weights=[0, 1]), [2, 5]
        )
        self.assertEqual(fusion.reciprocal_rank_fusion([[7], [2]], weights=[0, 0]), [])

    def test_duplicate_only_contributes_at_first_original_rank(self):
        # 1's first occurrence stays at rank 3; duplicates do not compress ranks.
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[9, 9, 1], [2, 1]]), [1, 2, 9]
        )
        self.assertEqual(fusion.reciprocal_rank_fusion([[9, 9, 1], [8, 2]]), [8, 9, 2, 1])
        self.assertEqual(fusion.reciprocal_rank_fusion([[9, 9, 1], [1]], k=1), [1, 9])

    def test_empty_individual_rankings_and_weight_alignment(self):
        self.assertEqual(fusion.reciprocal_rank_fusion([[], [9, 2], []]), [9, 2])
        self.assertEqual(
            fusion.reciprocal_rank_fusion([[], [9, 2], []], weights=[2, 0, 3]), []
        )

    def test_all_rankings_empty(self):
        for rankings in ([], [[]], [[], []]):
            with self.subTest(rankings=rankings):
                self.assertEqual(fusion.reciprocal_rank_fusion(rankings), [])

    def test_different_lengths(self):
        self.assertEqual(fusion.reciprocal_rank_fusion([[7, 2, 10], [2]]), [2, 7, 10])

    def test_exact_ties_use_numeric_id_independent_of_insertion_order(self):
        for rankings in ([[9, 2], [2, 9]], [[2, 9], [9, 2]], [[9], [2]], [[2], [9]]):
            with self.subTest(rankings=rankings):
                self.assertEqual(fusion.reciprocal_rank_fusion(rankings), [2, 9])

    def test_limits(self):
        for limit, expected in ((None, [7, 2, 10]), (2, [7, 2]), (9, [7, 2, 10]), (0, []), (-1, [])):
            with self.subTest(limit=limit):
                self.assertEqual(fusion.reciprocal_rank_fusion([[7, 2, 10]], limit=limit), expected)

    def test_invalid_weight_count(self):
        for weights in ([], [1], [1, 1, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fusion.reciprocal_rank_fusion([[1], []], weights=weights)

    def test_invalid_weights(self):
        for weight in (-1, float("nan"), float("inf"), -float("inf"), "1", None, True, 1j, {}, 10**1000):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                fusion.reciprocal_rank_fusion([[1]], weights=[weight])
        with self.assertRaises(ValueError):
            fusion.reciprocal_rank_fusion([[1]], weights=1)

    def test_invalid_k(self):
        for k in (0, -1, float("nan"), float("inf"), -float("inf"), "60", True, 1j, [], 10**1000):
            with self.subTest(k=k), self.assertRaises(ValueError):
                fusion.reciprocal_rank_fusion([[1]], k=k)

    def test_invalid_chunk_ids_including_zero_weight_and_zero_limit(self):
        for chunk_id in (-1, 1.0, "1", None, True, False, float("nan"), [], {}):
            for options in ({}, {"weights": [0]}, {"limit": 0}):
                with self.subTest(chunk_id=chunk_id, options=options), self.assertRaises(ValueError):
                    fusion.reciprocal_rank_fusion([[chunk_id]], **options)

    def test_inputs_are_not_mutated(self):
        rankings = [[7, 2, 7], [], [2, 5]]
        weights = [1, 0, 2]
        fusion.reciprocal_rank_fusion(rankings, weights=weights)
        self.assertEqual(rankings, [[7, 2, 7], [], [2, 5]])
        self.assertEqual(weights, [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
