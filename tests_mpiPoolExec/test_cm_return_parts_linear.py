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

if __name__ == '__main__': 
    unittest.main()