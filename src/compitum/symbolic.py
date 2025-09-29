from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SymbolicValue(ABC):
    """
    Abstract base class for a value that has both a symbolic (LaTeX)
    and a concrete (numerical) representation.
    """
    def __init__(self, name: str, value: Any):
        if not isinstance(name, str):
            raise TypeError("Symbolic name must be a string.")
        self.name = name
        self.value = value

    @abstractmethod
    def to_latex(self) -> str:
        """Return the LaTeX string representation of the value."""
        ...

    def evaluate(self) -> Any:
        """Return the concrete numerical value."""
        return self.value

    def __repr__(self) -> str:
        return f"SymbolicValue(name='{self.name}', value={self.value})"

    # --- Operator Overloading ---
    def __add__(self, other: SymbolicValue) -> SymbolicExpression:
        return SymbolicExpression(self, other, operator='+')

    def __mul__(self, other: SymbolicValue) -> SymbolicExpression:
        return SymbolicExpression(self, other, operator='*', latex_op=r' \cdot ')

    def __matmul__(self, other: SymbolicValue) -> SymbolicExpression:
        return SymbolicExpression(self, other, operator='@', latex_op='')


class SymbolicScalar(SymbolicValue):
    """Represents a scalar value."""
    def to_latex(self) -> str:
        """Return the LaTeX string representation of the scalar."""
        return self.name


class SymbolicMatrix(SymbolicValue):
    """Represents a matrix value."""
    def to_latex(self) -> str:
        """Return the LaTeX string representation of the matrix."""
        return self.name

    @property
    def T(self) -> SymbolicMatrix:
        """Returns a new SymbolicMatrix representing the transpose."""
        # This is a symbolic operation; the actual transpose happens during evaluation.
        return SymbolicMatrix(name=f"{self.name}^T", value=self.value.T)


class SymbolicExpression(SymbolicValue):
    """Represents a combination of two SymbolicValues via an operator."""
    def __init__(
        self,
        left: SymbolicValue,
        right: SymbolicValue,
        operator: str,
        latex_op: str | None = None
    ):
        self.left = left
        self.right = right
        self.operator = operator
        self.latex_op = latex_op if latex_op is not None else operator
        # The value of an expression is not known at creation, it must be evaluated.
        super().__init__(name=self.to_latex(), value=None)

    def to_latex(self) -> str:
        """Creates the LaTeX string for the expression."""
        # Basic formatting, can be improved for complex cases
        return f"({self.left.to_latex()} {self.latex_op} {self.right.to_latex()})"

    def evaluate(self) -> Any:
        """
        Evaluates the expression by applying the operator to the concrete
        values of the operands.
        """
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        if self.operator == '+':
            return left_val + right_val
        elif self.operator == '-':
            return left_val - right_val
        elif self.operator == '*':
            return left_val * right_val
        elif self.operator == '/':
            return left_val / right_val
        elif self.operator == '@':
            return left_val @ right_val
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
