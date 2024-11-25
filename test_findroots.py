import pytest
import math
import numpy as np
from findroots import func_1_safe, func_2, func_3, calculate_accuracy, apply_methods, plot_function_with_roots

# Test for func_1_safe
def test_func_1_safe():
    assert abs(func_1_safe(1)) - 1.557 < 1e-2
    with pytest.raises(ValueError):
        func_1_safe(math.pi)

# Test for func_2 (tanh)
def test_func_2():
    assert abs(func_2(0)) - 0.0 < 1e-6
    assert abs(func_2(1)) - 0.7616 < 1e-4
    assert abs(func_2(-1)) - (0.7616) < 1e-5

# Test for func_3 (sin)
def test_func_3():
    assert abs(func_3(0)) - 0.0 < 1e-6
    assert abs(func_3(math.pi / 2)) - 1.0 < 1e-6
    assert abs(func_3(math.pi)) - 0.0 < 1e-6

# Test for calculate_accuracy
def test_calculate_accuracy():
    assert calculate_accuracy(1.0, 1.0) == 0  # Exact match returns 0 accuracy
    assert calculate_accuracy(1.00001, 1.0) == 4

# Test for apply_methods
def test_apply_methods():
    try:
        apply_methods(func_2, (-1, 1), "y(x) = tanh(x)", true_root=0.0, filename="test_plot.png")
    except Exception as e:
        pytest.fail(f"apply_methods failed with error: {str(e)}")

# Test for plot_function_with_roots
def test_plot_function_with_roots():
    try:
        plot_function_with_roots(func_3, (0, math.pi), [math.pi], 'test_plot.png', "y(x) = sin(x)")
    except Exception as e:
        pytest.fail(f"plot_function_with_roots failed with error: {str(e)}")

# Test for find_roots
def test_find_roots():
    try:
        from findroots import find_roots
        find_roots()
    except Exception as e:
        pytest.fail(f"find_roots failed with error: {str(e)}")
