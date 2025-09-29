
from typing import Any

import numpy as np
import pytest

from compitum.symbolic import (
    SymbolicExpression,
    SymbolicMatrix,
    SymbolicScalar,
)


def test_symbolic_value_raises_type_error() -> None:
    """Test that SymbolicValue raises a TypeError if the name is not a string."""
    with pytest.raises(TypeError):
        bad_name: Any = 123
        SymbolicScalar(name=bad_name, value=456)


def test_symbolic_value_repr() -> None:
    """Test the __repr__ method of SymbolicValue."""
    sv = SymbolicScalar(name="a", value=1)
    assert repr(sv) == "SymbolicValue(name='a', value=1)"


def test_symbolic_expression_subtraction() -> None:
    """Test the subtraction operator for SymbolicExpression."""
    a = SymbolicScalar(name="a", value=5)
    b = SymbolicScalar(name="b", value=3)
    expr = SymbolicExpression(a, b, operator="-")
    assert expr.evaluate() == 2


def test_symbolic_expression_division() -> None:
    """Test the division operator for SymbolicExpression."""
    a = SymbolicScalar(name="a", value=6)
    b = SymbolicScalar(name="b", value=3)
    expr = SymbolicExpression(a, b, operator="/")
    assert expr.evaluate() == 2


def test_symbolic_expression_unknown_operator() -> None:
    """Test that SymbolicExpression raises a ValueError for an unknown operator."""
    a = SymbolicScalar(name="a", value=1)
    b = SymbolicScalar(name="b", value=2)
    expr = SymbolicExpression(a, b, operator="^")
    with pytest.raises(ValueError):
        expr.evaluate()

def test_symbolic_matrix_transpose_evaluation() -> None:
    """Test the evaluation of a transposed SymbolicMatrix."""
    m_val = np.array([[1, 2], [3, 4]])
    m = SymbolicMatrix(name="M", value=m_val)
    mT = m.T
    assert np.array_equal(mT.evaluate(), m_val.T)

def test_symbolic_expression_to_latex() -> None:
    """Test the to_latex method of SymbolicExpression."""
    a = SymbolicScalar(name="a", value=1)
    b = SymbolicScalar(name="b", value=2)
    expr = a + b
    assert expr.to_latex() == "(a + b)"
