from mpi4py import MPI 
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

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
# @pytest.mark.mpiexec(n=2)
# def run_test_cm_data_is_binary_evenly_distributed(nprocs):

@pytest.mark.mpiexec
@pytest.mark.parametrize("nprocs", [2])
def run_test_cm_data_is_binary_evenly_distributed(nprocs):
    assert MPI.COMM_WORLD.size == nprocs

# def test_cm_data_is_binary_evenly_distributed():
#     # Prepare
#     np.random.seed(0)

#     # two features with a quadratic relationship
#     feature0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
#     feature1 = np.random.rand(10)

#     # Run
#     cm_value, max_parts, parts = ccc(
#         feature0, feature1, internal_n_clusters=[2], return_parts=True
#     )

#     # Validate
#     assert cm_value < 0.05

#     assert parts is not None
#     assert len(parts) == 2
#     assert parts[0].shape == (1, 10)

#     # the partition should separate true from false values in data
#     assert ari(parts[0][0], feature0) == 1.0
