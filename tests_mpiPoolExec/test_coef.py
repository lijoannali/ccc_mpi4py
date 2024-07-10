# This file is for profiling and determining which functions take the most time to execute 
import cProfile
import re
import unittest

from concurrent.futures import ThreadPoolExecutor
from random import shuffle
import time
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_rand_score as ari

from ccc.coef import (
    ccc,
    get_range_n_clusters,
    run_quantile_clustering,
    get_perc_from_k,
    get_parts,
    get_coords_from_index,
    cdist_parts_basic,
    cdist_parts_parallel,
    get_chunks,
)


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

class TestCoef(unittest.TestCase):
    def test_cm_return_parts_linear(self):
        # Prepare
        np.random.seed(0)

        # two features on 100 objects with a linear relationship
        feature0 = np.random.rand(100)
        feature1 = feature0 * 5.0

        # Run
        cm_value, max_parts, parts = ccc(feature0, feature1, return_parts=True)

        # Validate
        self.assertEqual(cm_value, 1.0)
        
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (9, 100))
        self.assertEqual(parts[1].shape, (9, 100))
        
        self.assertIsNotNone(max_parts)
        self.assertTrue(hasattr(max_parts, "shape"))
        self.assertEqual(max_parts.shape, (2,))
        np.testing.assert_array_equal(max_parts, np.array([0, 0]))


    def test_get_perc_from_k_with_k_less_than_two(self):
        self.assertEqual(get_perc_from_k(1), [])
        self.assertEqual(get_perc_from_k(0), [])
        self.assertEqual(get_perc_from_k(-1), [])

    def test_get_perc_from_k(self):
        self.assertEqual(get_perc_from_k(2), [0.5])
        self.assertAlmostEqual(np.round(get_perc_from_k(3), 3).tolist(), [0.333, 0.667])
        self.assertEqual(get_perc_from_k(4), [0.25, 0.50, 0.75])

    def test_run_quantile_clustering_with_two_clusters01(self):
        #Prepare
        np.random.seed(0)
        data = np.concatenate((np.random.normal(0, 1, 10), np.random.normal(5, 1, 10)))
        data_ref = np.concatenate(([0] * 10, [1] * 10))
        idx_shuffled = list(range(len(data)))
        np.random.shuffle(idx_shuffled)
        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]
        part = run_quantile_clustering(data, 2)
        self.assertIsNotNone(part)
        self.assertEqual(len(part), 20)
        self.assertEqual(len(np.unique(part)), 2)
        self.assertEqual(ari(data_ref, part), 1.0)

    def test_run_quantile_clustering_with_two_clusters_mixed(self):
        np.random.seed(0)
        data = np.concatenate((
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        ))
        data_ref = np.concatenate(([0] * 10, [1] * 10))
        idx_shuffled = list(range(len(data)))
        np.random.shuffle(idx_shuffled)
        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]
        part = run_quantile_clustering(data, 2)
        self.assertIsNotNone(part)
        self.assertEqual(len(part), 20)
        self.assertEqual(len(np.unique(part)), 2)
        self.assertEqual(ari(data_ref, part), 1.0)

    def test_run_quantile_clustering_with_four_clusters(self):
        np.random.seed(0)
        data = np.concatenate((
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        ))
        data_ref = np.concatenate(([0] * 5, [1] * 5, [2] * 5, [3] * 5))
        idx_shuffled = list(range(len(data)))
        np.random.shuffle(idx_shuffled)
        data = data[idx_shuffled]
        data_ref = data_ref[idx_shuffled]
        part = run_quantile_clustering(data, 4)
        self.assertIsNotNone(part)
        self.assertEqual(len(part), 20)
        self.assertEqual(len(np.unique(part)), 4)
        self.assertEqual(ari(data_ref, part), 1.0)

class TestGetRangeNClusters(unittest.TestCase):

    def test_get_range_n_clusters_without_internal_n_clusters(self):
        range_n_clusters = get_range_n_clusters(100)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

        range_n_clusters = get_range_n_clusters(25)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))

    def test_get_range_n_clusters_with_internal_n_clusters_is_list(self):
        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[2])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[2, 3, 4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

    def test_get_range_n_clusters_with_internal_n_clusters_none(self):
        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=None)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=None)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))

    def test_get_range_n_clusters_with_internal_n_clusters_has_single_int(self):
        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[3])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([3]))

        range_n_clusters = get_range_n_clusters(5, internal_n_clusters=[4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([4]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[1])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

        range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[25])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

    def test_get_range_n_clusters_with_internal_n_clusters_are_less_than_two(self):
        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, 4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 1])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3]))

        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 0, 4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, -4, 6])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 6]))

    def test_get_range_n_clusters_with_internal_n_clusters_are_repeated(self):
        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2, 3, 2, 4])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

        range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2, 2, 2])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    def test_get_range_n_clusters_with_very_few_features(self):
        range_n_clusters = get_range_n_clusters(3)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2]))

        range_n_clusters = get_range_n_clusters(2)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

        range_n_clusters = get_range_n_clusters(1)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

        range_n_clusters = get_range_n_clusters(0)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

    def test_get_range_n_clusters_with_larger_k_than_features(self):
        range_n_clusters = get_range_n_clusters(10, internal_n_clusters=[10])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

        range_n_clusters = get_range_n_clusters(10, internal_n_clusters=[11])
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([]))

    def test_get_range_n_clusters_with_default_max_k(self):
        range_n_clusters = get_range_n_clusters(200)
        self.assertIsNotNone(range_n_clusters)
        np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_cm_basic(self):
        np.random.seed(123)
        feature0 = np.random.rand(100)
        feature1 = np.random.rand(100)
        cm_value = ccc(feature0, feature1)
        self.assertIsNotNone(cm_value)
        self.assertIsInstance(cm_value, float)
        self.assertAlmostEqual(cm_value, 0.01, delta=0.01)

    def test_cm_basic_internal_n_clusters_is_integer(self):
        np.random.seed(123)
        feature0 = np.random.rand(100)
        feature1 = np.random.rand(100)
        cm_value = ccc(feature0, feature1)
        self.assertIsNotNone(cm_value)
        self.assertIsInstance(cm_value, float)
        self.assertGreater(cm_value, 0.0)
        cm_value2 = ccc(feature0, feature1, internal_n_clusters=10)
        self.assertAlmostEqual(cm_value, cm_value2)

    def test_cm_basic_internal_n_clusters_is_integer_more_checks(self):
        np.random.seed(123)
        feature0 = np.random.rand(100)
        feature1 = np.random.rand(100)
        cm_value = ccc(feature0, feature1, internal_n_clusters=[2, 3, 4])
        self.assertIsNotNone(cm_value)
        self.assertIsInstance(cm_value, float)
        self.assertGreater(cm_value, 0.0)
        cm_value2 = ccc(feature0, feature1, internal_n_clusters=4)
        self.assertAlmostEqual(cm_value, cm_value2)

    def test_cm_ari_is_negative(self):
        feature0 = np.array([1, 2, 3, 4, 5])
        feature1 = np.array([2, 4, 1, 3, 5])
        cm_value = ccc(feature0, feature1)
        self.assertEqual(cm_value, 0.0)

    def test_cm_random_data(self):
        rs = np.random.RandomState(123)
        for i in range(10):
            feature0 = np.interp(rs.rand(100), (0, 1), (-1.0, 1.0))
            feature1 = rs.rand(100)
            cm_value = ccc(feature0, feature1)
            self.assertAlmostEqual(cm_value, 0.025, delta=0.025)

    def test_cm_linear(self):
        np.random.seed(0)
        feature0 = np.random.rand(100)
        feature1 = feature0 * 5.0
        cm_value = ccc(feature0, feature1)
        self.assertEqual(cm_value, 1.0)

    def test_cm_quadratic(self): #checked
        np.random.seed(1)
        feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
        feature1 = np.power(feature0, 2.0)
        cm_value = ccc(feature0, feature1)
        self.assertGreater(cm_value, 0.40)

    def test_cm_quadratic_noisy(self): #checked
        np.random.seed(1)
        feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
        feature1 = np.power(feature0, 2.0) + (0.10 * np.random.rand(feature0.shape[0]))
        cm_value = ccc(feature0, feature1)
        self.assertGreater(cm_value, 0.40)

    def test_cm_one_feature_with_all_same_values(self): #checked
        np.random.seed(0)
        feature0 = np.random.rand(100)
        feature1 = np.array([5] * feature0.shape[0])
        cm_value = ccc(feature0, feature1)
        self.assertTrue(np.isnan(cm_value))

    def test_cm_all_features_with_all_same_values(self): #checked
        np.random.seed(0)
        feature0 = np.array([0] * 100)
        feature1 = np.array([5] * feature0.shape[0])
        cm_value = ccc(feature0, feature1)
        self.assertTrue(np.isnan(cm_value))

    def test_cm_single_argument_is_matrix(self): #checked
        np.random.seed(0)
        feature0 = np.random.rand(100)
        feature1 = feature0 * 5.0
        feature2 = np.random.rand(feature0.shape[0])
        input_data = np.array([feature0, feature1, feature2])
        cm_value = ccc(input_data)
        self.assertIsNotNone(cm_value)
        self.assertTrue(hasattr(cm_value, "shape"))
        self.assertEqual(cm_value.shape, (3,))
        self.assertAlmostEqual(cm_value[0], 1.0)
        self.assertLess(cm_value[1], 0.03)
        self.assertLess(cm_value[2], 0.03)

    def test_cm_x_y_are_pandas_series(self): #checked
        np.random.seed(123)
        feature0 = pd.Series(np.random.rand(100))
        feature1 = pd.Series(np.random.rand(100))
        cm_value = ccc(feature0, feature1)
        self.assertIsNotNone(cm_value)
        self.assertIsInstance(cm_value, float)

    def test_cm_x_and_y_are_pandas_dataframe(self): #checked
        x = pd.DataFrame(np.random.rand(10, 100))
        y = pd.DataFrame(np.random.rand(10, 100))
        with self.assertRaises(ValueError) as e:
            ccc(x, y)
        self.assertIn("wrong combination", str(e.exception).lower())

    def test_cm_integer_overflow_random(self): #checked
        np.random.seed(0)
        feature0 = np.random.rand(1000000)
        feature1 = np.random.rand(1000000)
        cm_value = ccc(feature0, feature1)
        self.assertGreaterEqual(cm_value, 0.0)
        self.assertLessEqual(cm_value, 0.01)

    def test_cm_integer_overflow_perfect_match(self): #checked
        np.random.seed(0)
        feature0 = np.random.rand(1000000)
        cm_value = ccc(feature0, feature0)
        self.assertEqual(cm_value, 1.0)

    def test_get_parts_simple(self): #checked
        np.random.seed(0)
        feature0 = np.random.rand(100)
        parts = get_parts(feature0, (2,))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(np.unique(parts[0])), 2)

        parts = get_parts(feature0, (2, 3))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(np.unique(parts[0])), 2)
        self.assertEqual(len(np.unique(parts[1])), 3)

    def test_get_parts_with_singletons(self): #checked
        np.random.seed(0)
        feature0 = np.array([1.3] * 10)
        parts = get_parts(feature0, (2,))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 1)
        np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))

        parts = get_parts(feature0, (2, 3))
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))
        np.testing.assert_array_equal(np.unique(parts[1]), np.array([-2]))

    def test_get_parts_with_categorical_feature(self): #checked
        np.random.seed(0)
        feature0 = np.array([4] * 10)
        parts = get_parts(feature0, (2,), data_is_numerical=False)
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 1)
        np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))

        parts = get_parts(feature0, (2, 3), data_is_numerical=False)
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))
        np.testing.assert_array_equal(np.unique(parts[1]), np.array([-1]))

    def test_cdist_parts_one_vs_one(self): #checked
        from scipy.spatial.distance import cdist
        from sklearn.metrics import adjusted_rand_score as ari

        parts0 = np.array(
            [
                [1, 1, 2, 2, 3, 3],
            ]
        )
        parts1 = np.array(
            [
                [3, 3, 1, 1, 2, 2],
            ]
        )

        expected_cdist = cdist(parts0, parts1, metric=ari)
        np.testing.assert_array_equal(expected_cdist, np.array([[1.0]]))

        # basic version (one thread)
        observed_cdist = cdist_parts_basic(parts0, parts1)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with one thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with two threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)


    def test_cdist_parts_one_vs_one_dissimilar(self): #checked
        from scipy.spatial.distance import cdist
        from sklearn.metrics import adjusted_rand_score as ari
        parts0 = np.array(
        [
            [1, 1, 2, 1, 3, 3],
        ]
        )
        parts1 = np.array(
            [
                [3, 3, 1, 1, 2, 3],
            ]
        )
        expected_cdist = cdist(parts0, parts1, metric=ari)
        observed_cdist = cdist_parts_basic(parts0, parts1)
        np.testing.assert_array_equal(expected_cdist, np.array([[-0.022727272727272728]]))

        # basic version (one thread)
        observed_cdist = cdist_parts_basic(parts0, parts1)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with one thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with two threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

    def test_cdist_parts_one_vs_two(self): #checked
        from scipy.spatial.distance import cdist
        from sklearn.metrics import adjusted_rand_score as ari
        parts0 = np.array([[1, 1, 2, 1, 3, 3]])
        parts1 = np.array([[3, 3, 1, 1, 2, 3], [3, 3, 1, 1, 2, 2]])
        expected_cdist = cdist(parts0, parts1, metric=ari)
        np.testing.assert_array_equal(
        expected_cdist,
        np.array(
            [
                [-0.022727272727272728, 0.4444444444444444],
            ]
        ),
        )

        #basic version (one thread)
        observed_cdist = cdist_parts_basic(parts0, parts1)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with one thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

        # with two threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

    def test_cdist_parts_two_vs_two(self): #checked 
        from scipy.spatial.distance import cdist 
        from sklearn.metrics import adjusted_rand_score as ari 
        parts0 = np.array(
            [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 1, 3, 3]
            ]
        )
        parts1 = np.array(
            [
                [3, 3, 1, 1, 2, 3], 
                [3, 3, 1, 1, 2, 2]
            ]
        )

        expected_cdist = cdist(parts0, parts1, metric=ari)
        np.testing.assert_array_equal(
            expected_cdist,
            np.array(
                [
                    [0.4444444444444444, 1.0],
                    [-0.022727272727272728, 0.4444444444444444],
                ]
            ),
        )
        #basic version (one thread)
        observed_cdist = cdist_parts_basic(parts0, parts1)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)
        # with one thread 
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)
        # with two threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
        np.testing.assert_array_equal(observed_cdist, expected_cdist)

     def test_get_coords_from_index(self): #checked
        # data is an example with n_obj = 5 just to illustrate
        # data = np.array(
        #     [
        #         [10, 11],
        #         [23, 22],
        #         [27, 26],
        #         [37, 36],
        #         [47, 46],
        #     ]
        # )
        n_obj = 5

        res = get_coords_from_index(n_obj, 0)
        self.assertEqual(res, (0, 1))

        res = get_coords_from_index(n_obj, 1)
        self.assertEqual(res, (0, 2))

        res = get_coords_from_index(n_obj, 3)
        self.assertEqual(res, (0, 4))

        res = get_coords_from_index(n_obj, 4)
        self.assertEqual(res, (1, 2))

        res = get_coords_from_index(n_obj, 9)
        self.assertEqual(res, (3, 4))

    def test_get_coords_from_index_smaller(self):
        # data is an example with n_obj = 5 just to illustrate
        # data = np.array(
        #     [
        #         [10, 11],
        #         [23, 22],
        #         [27, 26],
        #         [37, 36],
        #     ]
        # )
        n_obj = 4

        res = get_coords_from_index(n_obj, 0)
        self.assertEqual(res, (0, 1))

        res = get_coords_from_index(n_obj, 1)
        self.assertEqual(res, (0, 2))

        res = get_coords_from_index(n_obj, 2)
        self.assertEqual(res, (0, 3))

        res = get_coords_from_index(n_obj, 3)
        self.assertEqual(res, (1, 2))

        res = get_coords_from_index(n_obj, 5)
        self.assertEqual(res, (2, 3))

    def setUp(self):
        self.input_data_dir = Path(__file__).parent / "data"

    def test_cm_values_equal_to_original_implementation(self):
        data = pd.read_pickle(self.input_data_dir / "ccc-random_data-data.pkl")
        data = data.to_numpy()

        corr_mat = ccc(data, internal_n_clusters=list(range(2, 10 + 1)))

        expected_corr_matrix = pd.read_pickle(self.input_data_dir / "ccc-random_data-coef.pkl")
        expected_corr_matrix = expected_corr_matrix.to_numpy()
        expected_corr_matrix = expected_corr_matrix[np.triu_indices(expected_corr_matrix.shape[0], 1)]

        np.testing.assert_almost_equal(expected_corr_matrix, corr_mat)

    def test_cm_return_parts_quadratic(self):
        np.random.seed(0)

        feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
        feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

        cm_value, max_parts, parts = ccc(feature0, feature1, internal_n_clusters=[2, 3], return_parts=True)

        self.assertAlmostEqual(cm_value, 0.31, places=2)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (2, 10))
        self.assertEqual(len(np.unique(parts[0][0])), 2)
        self.assertEqual(len(np.unique(parts[0][1])), 3)
        self.assertEqual(parts[1].shape, (2, 10))
        self.assertEqual(len(np.unique(parts[1][0])), 2)
        self.assertEqual(len(np.unique(parts[1][1])), 3)

        self.assertIsNotNone(max_parts)
        self.assertTrue(hasattr(max_parts, "shape"))
        self.assertEqual(max_parts.shape, (2,))
        np.testing.assert_array_equal(max_parts, np.array([1, 0]))

    def test_cm_return_parts_linear(self):
        np.random.seed(0)

        feature0 = np.random.rand(100)
        feature1 = feature0 * 5.0

        cm_value, max_parts, parts = ccc(feature0, feature1, return_parts=True)

        self.assertEqual(cm_value, 1.0)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (9, 100))
        self.assertEqual(parts[1].shape, (9, 100))

        self.assertIsNotNone(max_parts)
        self.assertTrue(hasattr(max_parts, "shape"))
        self.assertEqual(max_parts.shape, (2,))
        np.testing.assert_array_equal(max_parts, np.array([0, 0]))

    def test_cm_return_parts_categorical_variable(self):
        np.random.seed(0)

        numerical_feature0 = np.random.rand(100)
        numerical_feature0_median = np.percentile(numerical_feature0, 50)

        categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.unicode_)
        categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
        categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"

        cm_value, max_parts, parts = ccc(numerical_feature0, categorical_feature1, return_parts=True)

        self.assertEqual(cm_value, 1.0)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)

        self.assertEqual(parts[0].shape, (9, 100))
        self.assertEqual(set(range(2, 10 + 1)), set(map(lambda x: np.unique(x).shape[0], parts[0])))

        self.assertEqual(parts[1].shape, (9, 100))
        self.assertEqual(np.unique(parts[1][0, :]).shape[0], 2)
        unique_in_rest = np.unique(parts[1][1:, :])
        self.assertEqual(unique_in_rest.shape[0], 1)
        self.assertEqual(unique_in_rest[0], -1)

        self.assertIsNotNone(max_parts)
        self.assertTrue(hasattr(max_parts, "shape"))
        self.assertEqual(max_parts.shape, (2,))
        np.testing.assert_array_equal(max_parts, np.array([0, 0]))

    def test_cm_return_parts_with_matrix_as_input(self):
        np.random.seed(0)

        feature0 = np.random.rand(100)
        feature1 = feature0 * 5.0
        X = pd.DataFrame({"feature0": feature0, "feature1": feature1})

        cm_value, max_parts, parts = ccc(X, return_parts=True)

        self.assertEqual(cm_value, 1.0)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (9, 100))
        self.assertEqual(parts[1].shape, (9, 100))

        self.assertIsNotNone(max_parts)
        self.assertTrue(hasattr(max_parts, "shape"))
        self.assertEqual(max_parts.shape, (2,))
        np.testing.assert_array_equal(max_parts, np.array([0, 0]))

    def test_cm_data_is_binary_evenly_distributed(self):
        np.random.seed(0)

        feature0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        feature1 = np.random.rand(10)

        cm_value, _, parts = ccc(feature0, feature1, internal_n_clusters=[2], return_parts=True)

        self.assertLess(cm_value, 0.05)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (1, 10))

        self.assertEqual(ari(parts[0][0], feature0), 1.0)

    def test_cm_data_is_binary_not_evenly_distributed(self):
        np.random.seed(0)

        feature0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        feature1 = np.random.rand(10)

        cm_value, _, parts = ccc(feature0, feature1, internal_n_clusters=[2], return_parts=True)

        self.assertLess(cm_value, 0.05)

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (1, 10))

        self.assertEqual(ari(parts[0][0], feature0), 1.0)

    def test_cm_numerical_and_categorical_features_perfect_relationship(self):
        np.random.seed(123)

        numerical_feature0 = np.random.rand(100)
        numerical_feature0_median = np.percentile(numerical_feature0, 50)

        categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.unicode_)
        categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
        categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"

        cm_value = ccc(numerical_feature0, categorical_feature1)
        self.assertEqual(cm_value, 1.0)

        self.assertEqual(ccc(categorical_feature1, numerical_feature0), cm_value)

    def test_cm_numerical_and_categorical_features_strong_relationship(self):
        np.random.seed(123)

        numerical_feature0 = np.random.rand(100)
        numerical_feature0_perc = np.percentile(numerical_feature0, 25)

        categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.unicode_)
        categorical_feature1[numerical_feature0 < numerical_feature0_perc] = "l"
        categorical_feature1[numerical_feature0 >= numerical_feature0_perc] = "u"

        cm_value = ccc(numerical_feature0, categorical_feature1)

        self.assertGreater(cm_value, 0.3)

        self.assertLess(ccc(categorical_feature1, numerical_feature0), cm_value)

if __name__ == '__main__': 
    unittest.main()


     