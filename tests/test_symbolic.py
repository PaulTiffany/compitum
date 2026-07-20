from typing import Any

import numpy as np
import pytest

from compitum.symbolic import (
    SymbolicExpression,
    SymbolicMatrix,
    SymbolicScalar,
    SymbolicValue,
)


def test_symbolic_value_raises_type_error() -> None:
    """Test that SymbolicValue raises a TypeError if the name is not a string."""
    with pytest.raises(TypeError) as exc_info:
        bad_name: Any = 123
        SymbolicScalar(name=bad_name, value=456)
    # `pytest.raises` alone doesn't check the message -- assert the exact text.
    assert str(exc_info.value) == "Symbolic name must be a string."


def test_symbolic_value_to_latex_is_abstract() -> None:
    """No test ever attempts to instantiate SymbolicValue directly -- if
    `to_latex` stopped being `@abstractmethod`, the class would no longer be
    abstract and this would succeed instead of raising."""
    with pytest.raises(TypeError):
        SymbolicValue(name="x", value=1)  # type: ignore[abstract]


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
    with pytest.raises(ValueError) as exc_info:
        expr.evaluate()
    assert str(exc_info.value) == "Unknown operator: ^"


def test_symbolic_matrix_transpose_evaluation() -> None:
    """Test the evaluation of a transposed SymbolicMatrix."""
    m_val = np.array([[1, 2], [3, 4]])
    m = SymbolicMatrix(name="M", value=m_val)
    mT = m.T
    assert np.array_equal(mT.evaluate(), m_val.T)
    # .evaluate() alone never exercises the transposed matrix's own name/label.
    assert mT.name == "M^T"


def test_symbolic_expression_to_latex() -> None:
    """Test the to_latex method of SymbolicExpression."""
    a = SymbolicScalar(name="a", value=1)
    b = SymbolicScalar(name="b", value=2)
    expr = a + b
    assert expr.to_latex() == "(a + b)"


def test_symbolic_expression_addition_evaluates() -> None:
    """No existing test evaluates a __add__-built expression -- to_latex above
    only checks the LaTeX string, never the arithmetic result."""
    a = SymbolicScalar(name="a", value=5)
    b = SymbolicScalar(name="b", value=3)
    assert (a + b).evaluate() == 8


def test_symbolic_expression_multiplication_evaluates() -> None:
    """__mul__ was never exercised at all -- neither its evaluate() result
    nor its custom latex_op (" \\cdot ")."""
    a = SymbolicScalar(name="a", value=5)
    b = SymbolicScalar(name="b", value=3)
    expr = a * b
    assert expr.evaluate() == 15
    assert expr.to_latex() == r"(a  \cdot  b)"


def test_symbolic_expression_matmul_evaluates() -> None:
    """__matmul__ was never exercised at all."""
    m1 = SymbolicMatrix(name="M1", value=np.array([[1, 2], [3, 4]]))
    m2 = SymbolicMatrix(name="M2", value=np.array([[5, 6], [7, 8]]))
    expr = m1 @ m2
    assert np.array_equal(expr.evaluate(), m1.value @ m2.value)
    # .evaluate() alone never exercises __matmul__'s own latex_op="" (no
    # spacing/operator text between operands, unlike + or *).
    assert expr.to_latex() == "(M1  M2)"
