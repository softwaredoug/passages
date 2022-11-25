import numpy as np


# Assuming this math I did a long time ago is right
# https://github.com/o19s/hangry/blob/master/src/main/java/com/o19s/hangry/randproj/VectorUtils.java#L100
# You can find a projection (p1...p2) that divides two vectors a and b
# by solving the inequalities
#
#
# a1 * p1 + a2 * p2 ... + an * pn >= 0
# b1 * p1 + b2 * b2 ... + bn * pn < 0
#
# an * pn >= -(a1 * p1 + a2 * p2 + ... )
# bn * pn <  -(b1 * p1 + b2 * p2 + ... )
#
# an is positive:
# pn >= -(a1 * p1 + a2 * p2 + ... ) / an

# an is negative sign flips:
# pn < -(a1 * p1 + a2 * p2 + ... ) / an
#
# bn has different sign, it will also be
# pn < -(b1 * p1 + b2 * p2 + ... ) / bn
#
#
def projection_between(vect1: np.ndarray, vect2: np.ndarray):

    projection = np.random.random_sample(size=vect1.shape)
    projection /= np.linalg.norm(projection)

    dot1 = np.dot(projection[:-1], vect1[:-1])
    dot2 = np.dot(projection[:-1], vect2[:-1])

    # To get the floor / ceiling we solve an inequality
    # whose sign flips depending on whether vectn[-1] is
    # positive or negative
    if np.sign(vect1[-1]) == np.sign(vect2[-1]):
        floor = -dot1 / vect1[-1]
        ceiling = -dot2 / vect2[-1]
        projection[-1] = (floor + ceiling) / 2.0
    # Sign is different, then we can just get a large or
    # small number
    else:
        projection[-1] = -10

    projection /= np.linalg.norm(projection)
    return projection


def test_projection_random_positive():
    for i in range(0, 100):
        vect1 = np.random.random_sample(size=1000)
        vect2 = np.random.random_sample(size=1000)
        vect1 /= np.linalg.norm(vect1)
        vect2 /= np.linalg.norm(vect2)
        projection = projection_between(vect1, vect2)
        dot1 = np.dot(vect1, projection)
        dot2 = np.dot(vect2, projection)
        if dot1 > 0:
            assert dot2 < 0
        elif dot1 < 0:
            assert dot2 > 0


def test_projection_random_negative():
    for i in range(0, 100):
        vect1 = -np.random.random_sample(size=1000)
        vect2 = -np.random.random_sample(size=1000)
        vect1 /= np.linalg.norm(vect1)
        vect2 /= np.linalg.norm(vect2)
        projection = projection_between(vect1, vect2)
        dot1 = np.dot(vect1, projection)
        dot2 = np.dot(vect2, projection)
        if dot1 > 0:
            assert dot2 < 0
        elif dot1 < 0:
            assert dot2 > 0


def test_projection_random_opposite():
    for i in range(0, 100):
        vect1 = np.random.random_sample(size=1000)
        vect2 = -np.random.random_sample(size=1000)
        vect1 /= np.linalg.norm(vect1)
        vect2 /= np.linalg.norm(vect2)
        projection = projection_between(vect1, vect2)
        dot1 = np.dot(vect1, projection)
        dot2 = np.dot(vect2, projection)
        if dot1 > 0:
            assert dot2 < 0
        elif dot1 < 0:
            assert dot2 > 0


def test_projection_between():
    vect1 = np.array([0.707, 0.707])
    vect2 = np.array([0.9, 0.1])
    vect1 /= np.linalg.norm(vect1)
    vect2 /= np.linalg.norm(vect2)

    projection = projection_between(vect1, vect2)
    print(projection)
    dot1 = np.dot(vect1, projection)
    dot2 = np.dot(vect2, projection)
    if dot1 > 0:
        assert dot2 < 0
    elif dot1 < 0:
        assert dot2 > 0


def test_projection_between_opposite_sign2():
    vect1 = np.array([0.707, 0.707])
    vect2 = np.array([0.9, -0.1])
    vect1 /= np.linalg.norm(vect1)
    vect2 /= np.linalg.norm(vect2)

    projection = projection_between(vect1, vect2)
    print(projection)
    dot1 = np.dot(vect1, projection)
    dot2 = np.dot(vect2, projection)
    if dot1 > 0:
        assert dot2 < 0
    elif dot1 < 0:
        assert dot2 > 0


def test_projection_between_opposite_sign1():
    vect1 = np.array([0.707, -0.707])
    vect2 = np.array([0.9, 0.1])

    projection = projection_between(vect1, vect2)
    print(projection)
    dot1 = np.dot(vect1, projection)
    dot2 = np.dot(vect2, projection)
    if dot1 > 0:
        assert dot2 < 0
    elif dot1 < 1:
        assert dot2 > 0


def test_projection_between_sign():
    vect1_src = np.array([0.707, 0.707])
    vect2_src = np.array([0.9, 0.1])

    signs = [np.array([-1, 1]),
             np.array([1, -1]),
             np.array([-1, -1])]

    for i in range(0, 6):
        print(i)

        vect1 = vect1_src.copy()
        vect2 = vect2_src.copy()

        if i < 3:
            vect1 *= signs[i]
        else:
            vect2 *= signs[i // 2]

        projection = projection_between(vect1, vect2)
        print(projection)
        dot1 = np.dot(vect1, projection)
        dot2 = np.dot(vect2, projection)
        if dot1 > 0:
            assert dot2 < 0
        elif dot1 < 1:
            assert dot2 > 0
