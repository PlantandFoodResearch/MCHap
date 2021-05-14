import numpy as np

from mchap.assemble import arraymap


def test_get__nan():
    amap = arraymap.new(5, 2)
    for _ in range(10):
        a = np.random.randint(0, 2, 5)
        actual = arraymap.get(amap, a)
        assert np.isnan(actual)


def test_get_set_get():
    amap = arraymap.new(5, 3)
    a = np.array([0, 1, 2, 0, 1])
    b = np.array([0, 1, 2, 0, 2])
    assert np.isnan(arraymap.get(amap, a))
    assert np.isnan(arraymap.get(amap, b))
    amap = arraymap.set(amap, a, 0.5)
    assert arraymap.get(amap, a) == 0.5
    assert np.isnan(arraymap.get(amap, b))
    amap = arraymap.set(amap, b, 0.1)
    assert arraymap.get(amap, a) == 0.5
    assert arraymap.get(amap, b) == 0.1


def test_set__grow_tree():
    amap = arraymap.new(5, 3, initial_size=4)
    a = np.array([0, 1, 2, 0, 1])
    b = np.array([0, 1, 2, 0, 2])
    assert np.isnan(arraymap.get(amap, a))
    assert np.isnan(arraymap.get(amap, b))
    assert amap[0].shape == (4, 3)
    assert amap[1].shape == (4,)
    amap = arraymap.set(amap, a, 0.5)
    assert arraymap.get(amap, a) == 0.5
    assert np.isnan(arraymap.get(amap, b))
    assert amap[0].shape == (8, 3)
    assert amap[1].shape == (4,)
    amap = arraymap.set(amap, b, 0.1)
    assert arraymap.get(amap, a) == 0.5
    assert arraymap.get(amap, b) == 0.1
    assert amap[0].shape == (16, 3)
    assert amap[1].shape == (4,)


def test_set__grow_values():
    amap = arraymap.new(5, 3, initial_size=2)
    a = np.array([0, 1, 2, 0, 1])
    b = np.array([0, 1, 2, 0, 2])
    assert np.isnan(arraymap.get(amap, a))
    assert np.isnan(arraymap.get(amap, b))
    assert amap[0].shape == (2, 3)
    assert amap[1].shape == (2,)
    amap = arraymap.set(amap, a, 0.5)
    assert arraymap.get(amap, a) == 0.5
    assert np.isnan(arraymap.get(amap, b))
    assert amap[0].shape == (8, 3)
    assert amap[1].shape == (4,)
    amap = arraymap.set(amap, b, 0.1)
    assert arraymap.get(amap, a) == 0.5
    assert arraymap.get(amap, b) == 0.1
    assert amap[0].shape == (16, 3)
    assert amap[1].shape == (4,)
