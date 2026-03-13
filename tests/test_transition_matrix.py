import pytest
import numpy as np

from autoprot import transitions
from autoprot.utils import pack_vec

# def test_calc_tmatrix():
#     state_vecs = np.array([
#         [0,0],
#         [0,1],
#         [1,0],
#         [1,1]
#     ])
#     state_strs = [pack_vec(state_vec) for state_vec in state_vecs]
#     ps_all = np.array([

#     ])