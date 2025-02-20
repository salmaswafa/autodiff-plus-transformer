from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch

device = torch.device("cpu")
print(f"Using device: {device}")

class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return torch.mul(input_values[0], input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return torch.mul(input_values[0], node.constant)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]
    
class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class GreaterThanConstOp(Op):
    """Op to compare if node_A > Constant element-wise."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}>{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 1
        return (torch.gt(input_values[0], node.attrs["constant"])).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0])]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        # return input_values[0] - input_values[1]
        return torch.sub(input_values[0], input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]
    
class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """
    Op to compute sum along specified dimensions.
    
    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        result = input_values[0].sum(dim=node.dim, keepdim=node.keepdim)
        
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs['dim']
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]
        else:
            reshape_grad = expand_as_3d(output_grad, node.inputs[0])
            return [reshape_grad]

class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=0), zeros_like(output_grad)]
    
class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        # print('expand_op',input_tensor.shape, target_tensor.shape)
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=(0, 1)), zeros_like(output_grad)]

class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])
        # return torch.broadcast_to(input_values[0], target_shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.
        
        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")
            
        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]
        
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
                
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum)
            
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))))
            
        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        A, B = input_values[0], input_values[1]
        
        A_dims = len(A.shape)
        B_dims = len(B.shape)
        
        if A_dims == 2 and B_dims == 3:
            A = A.unsqueeze(1)
        
        result = torch.div(A, B)
        
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        return [div(output_grad, node.inputs[1]), (mul_by_const(div(mul(output_grad, node.inputs[0]), (power(node.inputs[1],2))), -1)) ]

class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return torch.div(input_values[0], node.attrs["constant"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [div_by_const(output_grad, node.attrs["constant"])]

class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.
        
        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        return torch.transpose(input_values[0], node.attrs["dim0"], node.attrs["dim1"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        # Gradient of transpose is transpose of the output gradient
        return [transpose(output_grad, node.attrs["dim0"], node.attrs["dim1"])]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return torch.matmul(input_values[0], input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Compute the gradient of the matrix multiplication operation.

        Parameters:
        ----------
        node: Node
            The MatMul node.
        output_grad: Node
            The gradient of the loss with respect to the output of the MatMul.

        Returns:
        -------
        List[Node]
            The gradient of the loss with respect to each input of the MatMul.
        """
        A, B = node.inputs

        # dL/dA = dL/dy @ dy/dA = output_grad @ B^T
        # dL/dy = output_grad
        # in this order given the dimensions of the matrices suitable for matmul
        grad_A = matmul(output_grad, transpose(B, -1, -2))

        # dL/dB = dy/dB @ dL/dy = output_grad @ B^T
        # dL/dy = output_grad
        # in this order given the dimensions of the matrices suitable for matmul
        grad_B = matmul(transpose(A, -1, -2), output_grad)

        return [grad_A, grad_B]
    

class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        
        x = input_values[0]
        dim = node.attrs["dim"]

        # Subtract the max value (numerical stability)
        x = x - torch.max(x, dim=dim, keepdim=True).values

        # Compute exponential of each element
        exp_x = torch.exp(x)

        # Compute the sum of the exponentials of all elements
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

        # Divide the exponential of each element with the sum of all exponentials (softmax probabilities)
        # All add up to 1
        softmax_probs = exp_x / sum_exp_x

        return softmax_probs


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        # gradient function with the jacobian trick:
        # softmax_probs×(output_grad−∑(output_grad×softmax_probs,dim,keepdim=True))
        
        x = node.inputs[0]
        dim = node.attrs["dim"]

        # Recompute softmax
        softmax_probs = softmax(x, dim=dim)  

        # Compute the weighted sum of the output gradients
        # ∑(output_grad×softmax_probs,dim,keepdim=True)
        sum_term = sum_op(mul(output_grad, softmax_probs), dim=dim, keepdim=True)

        # Apply the Jacobian trick to compute the gradient
        grad_wrt_input = mul(softmax_probs, sub(output_grad, sum_term))
        
        return [grad_wrt_input]

class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        x = input_values[0]
        return torch.where(x > 0, x, 0.0)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        return [mul(output_grad, greater_than_constant(node.inputs[0], 0))]

class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [div(output_grad, (mul_by_const(sqrt(node.inputs[0]), 2)))]

class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.pow(input_values[0], node.attrs["exponent"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        p = node.attrs["exponent"]
        return [mul(mul_by_const(output_grad, p), power(node.inputs[0], p-1))]

class MeanOp(Op):
    """Op to compute mean along specified dimensions.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.mean(input_values[0], node.attrs["dim"], node.attrs["keepdim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Compute the gradient of the Mean operation.

        Parameters:
        ----------
        node: Node
            The Mean node.
        output_grad: Node
            The gradient of the loss with respect to the output of the Mean.

        Returns:
        -------
        List[Node]
            The gradient of the loss with respect to the input of the Mean.
        """
        x = node.inputs[0]
        dim = node.attrs["dim"]
        keepdim = node.attrs.get("keepdim", False)

        # Count the number of elements in the reduced dimensions
        ones = ones_like(x)
        num_elements = sum_op(ones, dim=dim, keepdim=True)

        # Divide the output_grad evenly over the number of elements 
        grad_wrt_input = div(output_grad, num_elements)

        # If keepdim is False, expand the gradient to match the input shape
        if not keepdim:
            grad_wrt_input = mul(grad_wrt_input, ones)

        return [grad_wrt_input]

class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        
        x = input_values[0]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        
        # Normalized dimensions are the last ones (so operate on the last normalized_dims)
        # Usually the dims at the beginning are like batch_size, etc.
        normalized_dims = len(normalized_shape) # number of dims in the normalized shape
        normalized_axes = tuple(range(-normalized_dims, 0))

        # Compute mean and variance along the normalized dimensions
        mu = torch.mean(x, dim=normalized_axes, keepdim=True)
        var = torch.var(x, dim=normalized_axes, unbiased=False, keepdim=True)

        # Normalize
        # formula: x_hat = (x - mean) / sqrt(var + eps)
        numer = x - mu
        denom = torch.sqrt(var + eps)
        x_hat = torch.div(numer, denom)

        return x_hat

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Compute the gradient of the LayerNorm operation using the provided logic.

        Parameters:
        ----------
        node: Node
            The LayerNorm node.
        output_grad: Node
            The gradient of the loss with respect to the output of the LayerNorm.

        Returns:
        -------
        List[Node]
            The gradient of the loss with respect to the input of the LayerNorm.
        """
        
        # LOGIC
        # compute mean(x, dims) and var(x, dims)
        # compute mean(g, dims) and mean( (x - mu) * g, dims )
        # construct bracket term = g - mean(g) - ((x - mu) * mean((x - mu)*g) / (var + eps))
        # final gradient is bracket / sqrt(var + eps)
        
        x = node.inputs[0]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]

        # Normalized dimensions are the last ones (so operate on the right-most normalized_dims)
        # Usually the dims at the beginning are like batch_size, etc.
        normalized_dims = len(normalized_shape) # number of dims in the normalized shape
        normalized_axes = tuple(range(-normalized_dims, 0))

        # Compute mean and variance along the normalized dimensions (for "x")
        mu = mean(x, dim=normalized_axes, keepdim=True)  # mean(x, dims)
        var = mean(power(sub(x, mu), 2), dim=normalized_axes, keepdim=True)  # var(x, dims)

        # Compute mean(g, dims) and mean((x - mu) * g, dims)
        g = output_grad  # Gradient of the loss with respect to the output
        mean_g = mean(g, dim=normalized_axes, keepdim=True)  # mean(g, dims)
        mean_xmu_g = mean(mul(sub(x, mu), g), dim=normalized_axes, keepdim=True)  # mean((x - mu) * g, dims)

        # Construct the bracket term
        bracket_term = sub(
            sub(g, mean_g),  # g - mean(g)
            div(
                mul(sub(x, mu), mean_xmu_g),  # (x - mu) * mean((x - mu) * g)
                add_by_const(var, eps)  # var + eps
            )
        )

        # Final gradient
        grad_wrt_x = div(bracket_term, sqrt(add_by_const(var, eps)))

        return [grad_wrt_x]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
greater_than_constant = GreaterThanConstOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()

def topological_sort(nodes):
    """Helper function to perform topological sort on nodes.
    
    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort
        
    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    all_nodes = set() # unique set of all nodes in the graph
    in_degree = defaultdict(int) # keeps count of the in_degree of every node
    full_graph = defaultdict(list) # keeps track of the children of every node
    sorted_graph = [] # stores the nodes in sorted topological order
    
    # add node to data all_nodes, in_degree and full_graph
    # prep for sorting
    def process_node(node):
        if node not in all_nodes:
            # add new node
            all_nodes.add(node)
            
            # loop over all children
            for input_node in node.inputs:
                in_degree[node] += 1
                full_graph[input_node].append(node)
                process_node(input_node)
    
    # add every output node with its chain of inputs to the graph
    for node in nodes:
        process_node(node)
        
    # sorting

    # add nodes with in_degree == 0 first (inputs i.e. have no dependencies)
    q = deque([n for n in all_nodes if in_degree[n] == 0])
    
    while q:
        node = q.popleft()
        sorted_graph.append(node)
        
        # update dependencies and check if there are nodes without any dependencies now
        for child in full_graph[node]:
            in_degree[child] -= 1
            
            if in_degree[child] == 0:
                q.append(child)
    
    return sorted_graph    

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        sorted_nodes = topological_sort(self.eval_nodes)
        # print(sorted_nodes)
        
        # Process every node
        node_values = {}
        count = 0 # for debugging
        
        for node in sorted_nodes:
            if node.op == placeholder:
                # retrieve value from input_values
                node_values[node] = input_values[node]
            else:
                # compute value using the corresponding op
                input_values_list = [node_values[input_node] for input_node in node.inputs]
                
                # print(count)
                # print(f'op: {node.op}    =>     {node}')
                # for iv in (input_values_list):
                #     print(f'input shape: {iv.shape}')
                    
                # print(f'inputs: {input_values_list}')
                # print(f'op: {input}')
                node_values[node] = node.op.compute(node, input_values_list)
                # print(f'output shape: {node_values[node].shape}\n\n')
                count += 1
                
        return [node_values[node] for node in self.eval_nodes]

def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    # sort in topological order => for gradient we'll do reverse topological order
    sorted_nodes = topological_sort([output_node])
    # print(sorted_nodes)

    # Store gradients for every node
    grad_dict = {}
    grad_dict[output_node] = ones_like(output_node)  # gradient of output wrt itself is 1

    # Backpropagation => reverse order of nodes
    for node in reversed(sorted_nodes):

        # Compute gradients for input nodes using the node's operation
        if node.inputs:
            output_grad = grad_dict[node]
            # use op's gradient function to compute the gradient wrt its inputs
            input_grads = node.op.gradient(node, output_grad) 
 
            # Accumulate gradients for input nodes
            for input_node, input_grad in zip(node.inputs, input_grads):
                # if it exists, it means it is the input of multiple nodes, so we have to add all values together
                if input_node in grad_dict:
                    grad_dict[input_node] = grad_dict[input_node] + input_grad
                # if it doesn't exist, we just put this single gradient value
                else:
                    grad_dict[input_node] = input_grad

    # Return all gradients
    return [grad_dict[node] for node in nodes]

