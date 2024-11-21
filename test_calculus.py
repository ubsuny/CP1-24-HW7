"""
Unit testing module for testing functions in calculus.py
"""
import numpy as np
import calculus as calc

def test_dummy():
    """ 
    Unit test for dummy function
    """
    assert calc.dummy() == 0

def test_trapezoid_numpy():
    '''
    Unit test for numpy implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_numpy(np.sin, 0, np.pi), 2)

def test_trapezoid_scipy():
    '''
    Unit test for scipy implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_scipy(np.sin, 0, np.pi), 2)

def test_trapezoid_python():
    '''
    Unit test for pure python implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_python(np.sin, 0, np.pi), 2)
