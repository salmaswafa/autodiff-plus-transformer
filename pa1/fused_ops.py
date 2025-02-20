from typing import Any, Dict, List
import torch
import auto_diff as ad

class MatMulLayerNormOp(ad.Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: ad.Node, 
        node_B: ad.Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> ad.Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return ad.Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: ad.Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        A, B = input_values[0], input_values[1]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]

        # Matmul
        x = torch.matmul(A, B)
        
        # Layernorm
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

    def gradient(self, node: ad.Node, output_grad: ad.Node) -> List[ad.Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        # raise NotImplementedError
        
        A, B = node.inputs
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        
        # Step 1: Compute the gradient of the softmax
        x = ad.matmul(A, B)
        
        # Normalized dimensions are the last ones (so operate on the right-most normalized_dims)
        # Usually the dims at the beginning are like batch_size, etc.
        normalized_dims = len(normalized_shape) # number of dims in the normalized shape
        normalized_axes = tuple(range(-normalized_dims, 0))

        # Compute mean and variance along the normalized dimensions (for "x")
        mu = ad.mean(x, dim=normalized_axes, keepdim=True)  # mean(x, dims)
        var = ad.mean(ad.power(ad.sub(x, mu), 2), dim=normalized_axes, keepdim=True)  # var(x, dims)

        # Compute mean(g, dims) and mean((x - mu) * g, dims)
        g = output_grad  # Gradient of the loss with respect to the output
        mean_g = ad.mean(g, dim=normalized_axes, keepdim=True)  # mean(g, dims)
        mean_xmu_g = ad.mean(ad.mul(ad.sub(x, mu), g), dim=normalized_axes, keepdim=True)  # mean((x - mu) * g, dims)

        # Construct the bracket term
        bracket_term = ad.sub(
            ad.sub(g, mean_g),  # g - mean(g)
            ad.div(
                ad.mul(ad.sub(x, mu), mean_xmu_g),  # (x - mu) * mean((x - mu) * g)
                ad.add_by_const(var, eps)  # var + eps
            )
        )

        # Final gradient
        grad_wrt_x = ad.div(bracket_term, ad.sqrt(ad.add_by_const(var, eps)))

        # Step 2: Compute the gradient of the matrix multiplication
        grad_A = ad.matmul(grad_wrt_x, ad.transpose(B, -1, -2))
        grad_B = ad.matmul(ad.transpose(A, -1, -2), grad_wrt_x)
        
        return [grad_A, grad_B]


class MatMulSoftmaxOp(ad.Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: ad.Node, 
        node_B: ad.Node, 
        dim: int = -1
    ) -> ad.Node:
        return ad.Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: ad.Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        dim = node.attrs["dim"]
        
        # Matmul
        x = torch.matmul(input_values[0], input_values[1])

        # Softmax
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

    def gradient(self, node: ad.Node, output_grad: ad.Node) -> List[ad.Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A, B = node.inputs
        dim = node.attrs["dim"]

        # Matmul
        x = ad.matmul(A, B)

        # Recompute softmax
        softmax_probs = ad.softmax(x, dim=dim)  

        # Compute the weighted sum of the output gradients
        # ∑(output_grad×softmax_probs,dim,keepdim=True)
        sum_term = ad.sum_op(ad.mul(output_grad, softmax_probs), dim=dim, keepdim=True)

        # Apply the Jacobian trick to compute the gradient
        grad_wrt_input = ad.mul(softmax_probs, ad.sub(output_grad, sum_term))

        # Step 2: Compute the gradient of the matrix multiplication
        grad_A = ad.matmul(grad_wrt_input, ad.transpose(B, -1, -2))
        grad_B = ad.matmul(ad.transpose(A, -1, -2), grad_wrt_input)
        
        return [grad_A, grad_B]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()