import numpy as np
import pytest
import random

from random_projection import projection_between


def random_vector(sign: int = 1, dim=0):
    vect = np.random.random_sample(size=1000)
    vect /= np.linalg.norm(vect)
    vect[dim] *= sign
    return vect


def assert_projection_bisects(projection, vect1, vect2):
    dot1 = np.dot(vect1, projection)
    dot2 = np.dot(vect2, projection)
    if dot1 > 0:
        assert dot2 < 0
    elif dot1 < 0:
        assert dot2 > 0
    else:
        assert False, "Both cant be 0"


def test_projection_random_positive():
    for i in range(0, 100):
        vect1 = random_vector(sign=1)
        vect2 = random_vector(sign=1)
        projection = projection_between(vect1, vect2)
        assert_projection_bisects(projection, vect1, vect2)


def test_projection_random_negative():
    for i in range(0, 100):
        vect1 = random_vector(sign=-1)
        vect2 = random_vector(sign=-1)
        projection = projection_between(vect1, vect2)
        assert_projection_bisects(projection, vect1, vect2)


def test_projection_random_pos_neg():
    for i in range(0, 100):
        vect1 = random_vector(sign=1)
        vect2 = random_vector(sign=-1)
        projection = projection_between(vect1, vect2)
        assert_projection_bisects(projection, vect1, vect2)


def test_projection_random_neg_pos():
    for i in range(0, 100):
        vect1 = random_vector(sign=-1)
        vect2 = random_vector(sign=1)
        projection = projection_between(vect1, vect2)
        assert_projection_bisects(projection, vect1, vect2)


def test_projection_zeros_dim():
    vect1 = random_vector(sign=0)
    vect2 = random_vector(sign=1)
    with pytest.raises(ValueError):
        vect1 = random_vector(sign=0)
        vect2 = random_vector(sign=1)
        projection_between(vect1, vect2)

    with pytest.raises(ValueError):
        vect1 = random_vector(sign=1)
        vect2 = random_vector(sign=0)
        projection_between(vect1, vect2)


def test_projection_specify_dim():
    vect1 = random_vector(sign=1, dim=5)
    vect2 = random_vector(sign=-1, dim=5)
    projection = projection_between(vect1, vect2, dim=5)
    assert_projection_bisects(projection, vect1, vect2)


def test_projection_random_specify_dim():
    for i in range(0, 100):
        vect1 = random_vector(sign=1, dim=5)
        vect2 = random_vector(sign=1, dim=5)
        projection = projection_between(vect1, vect2, dim=5)
        assert_projection_bisects(projection, vect1, vect2)


def test_bad_case_due_to_not_using_custom_dims_correctly():
    random.seed(0)
    np.random.seed(0)

    vect1 = np.array([0.1487126, 0.683811, 0.27932906, -0.65746662])
    vect2 = np.array([0.55509923, 0.0941192, 0.68748675, 0.45865933])
    dim = 2
    projection = projection_between(vect1, vect2, dim=dim)
    assert_projection_bisects(projection, vect1, vect2)


def test_bad_case_due_to_identical_vectors():
    vect1 = np.array([0.58090311, 0.08887496, 0.76123888, -0.27416818])
    vect2 = np.array([0.58090311, 0.08887496, 0.76123888, -0.27416818])
    dim = 0
    with pytest.raises(ValueError):
        projection_between(vect1, vect2, dim=dim)
