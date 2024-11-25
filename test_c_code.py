import ctypes
import numpy as np
lib =ctypes.cdll.LoadLibrary("./c_code.dll")
dx=10
lib.adapt_c.argtypes=[ctypes.CFUNCTYPE(ctypes.c_double,ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
lib.adapt_c.restype=ctypes.c_double

def cubic(x):
    return x**3+1

def cosine(x):
    return np.cos(1/x)

def exponential(x):
    return np.exp(-1/x)
    
cubic_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_cubic_c=cubic_c(cubic)

cos_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_cos_c=cos_c(cosine)

exp_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_exp_c=exp_c(exponential)

def test_adapt():
    """
    test adapt confirms that the adapt_c function 
    produces the expected results for certain 
    integrals.
    """
    result1=lib.adapt_c(python_cubic_c, -1,1,10,10)
    assert np.isclose(result1, 2, .1)
    result2=lib.adapt_c(python_cos_c, .01,3*np.pi, 20,100)
    assert np.isclose(result2, 7.9, .1)
    result3=lib.adapt_c(python_exp_c, .01, 10, 100, 1000)
    assert np.isclose(result3, 7.2, .1)

    

