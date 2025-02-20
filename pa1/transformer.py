from typing import Callable, List

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms
import gc
import psutil

max_len = 28
torch.set_default_dtype(torch.float32)

device = torch.device("cpu")
print(f"Using device: {device}")

# General flow: 
# 0. "compute" fn of nodes actually computes the values! "gradient" fn of the nodes constructs the backward node
# 1. Construct forward graph
# 2. Construct backward graph
# 3. Input the data into this big graph (forward and backward)

def transformer(X: ad.Node, nodes: List[ad.Node], 
                model_dim: int, seq_length: int, eps: float, batch_size: int, num_classes: int) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.
    eps: float
        A small constant for numerical stability in LayerNorm.
    batch_size: int
        The size of the mini-batch.
    num_classes: int
        The number of output classes.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    # Weights
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # Matmul the querry weights, key weights and value weights with the input: X@W_Q, X@W_K, X@W_V
    Q = ad.matmul(X, W_Q)  # (batch_size, seq_length, model_dim)
    K = ad.matmul(X, W_K)  # (batch_size, seq_length, model_dim)
    V = ad.matmul(X, W_V)  # (batch_size, seq_length, model_dim)

    # Calculate attention: Query with Key
    scores = ad.matmul(Q, ad.transpose(K, dim0=-1, dim1=-2))  # QK^T (batch_size, seq_length, seq_length)
    d_k = model_dim ** 0.5  # Scaling factor
    scores = ad.div_by_const(scores, d_k)  # Scale by sqrt(d_k)
    A = ad.softmax(scores, dim=-1)  # Apply softmax over last dimension

    # Use the scaled dot product attention - with the values
    attention_output = ad.matmul(A, V)  # (batch_size, seq_length, model_dim)

    # Output transformation
    output_transform = ad.matmul(attention_output, W_O)  # (batch_size, seq_length, model_dim)

    # LayerNorm
    # Normalize over the last 2 dimensions (seq_length, model_dim)
    output_transform = ad.layernorm(output_transform, normalized_shape=[seq_length, model_dim], eps=eps)

    # Broadcast biases
    # Broadcast b_1 from (model_dim,) to (batch_size, seq_length, model_dim)
    b_1_broadcasted = ad.broadcast(b_1, input_shape=[model_dim], target_shape=[batch_size, seq_length, model_dim])
    # Broadcast b_2 from (num_classes,) to (batch_size, num_classes)
    b_2_broadcasted = ad.broadcast(b_2, input_shape=[num_classes], target_shape=[batch_size, num_classes])

    # Feedforward network
    hidden = ad.add(ad.matmul(output_transform, W_1), b_1_broadcasted)  # Linear transformation
    hidden = ad.relu(hidden)  # Activation
    
    # Average over sequence length
    # Average the hidden representation over the sequence length (dim=1)
    hidden_avg = ad.mean(hidden, dim=(1,), keepdim=False)  # (batch_size, model_dim)
    
    # Add another layernorm
    hidden_avg = ad.layernorm(hidden_avg, normalized_shape=[model_dim], eps=eps)

    # Output layer
    logits = ad.add(ad.matmul(hidden_avg, W_2), b_2_broadcasted)  # (batch_size, num_classes)

    return logits

def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node of shape (batch_size, num_classes), containing the logits for the batch of instances.

    y_one_hot: ad.Node
        A node of shape (batch_size, num_classes), containing the one-hot encoding of the ground truth label.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch, returning a zero-rank array (`()` shape).
    """

    # Function: 
    # Step 1 (softmax of logits): f(Z) = softmax(Z) 
    # Step 2 (cross-entropy loss): CE_Loss = -∑y_one_hot*log(f(Z))

    # Step 1
    # Use the implemented SoftmaxOp - dim = 1 is the num_classes (logits), dim = 0 is the batch
    softmax_probs = ad.softmax(Z, dim=1)  
    
    # Step 2
    total_loss = ad.mul_by_const(
                    ad.sum_op(
                        ad.mul(
                            y_one_hot, 
                            ad.log(softmax_probs)
                        ), 
                        dim=1, keepdim=False)  # Sum across classes
                    , -1)

    # Average loss over batch
    loss = ad.div_by_const(total_loss, batch_size) 

    return loss

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        # if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        logits, loss, *grads = f_run_model(X_batch, y_batch, model_weights)
        
        # print(f'grads.len: {len(grads)}')
        # print(f'grads: {grads}')
        # print(f'*grads: {*grads}')
        # print(*grads)
        
        # Update weights and biases
        for j in range(len(model_weights)):
            if grads[j] is None:
                raise ValueError(f"Gradient for model_weights[{j}] is None!")
            
            # print(f'grads[j]: {grads[j]}')
            # print(f'model_weights[j] before: {model_weights[j]}')
            # weight_change = ad.mul_by_const(grads[j], lr)  # W = W - lr * gradient
            
            # weight difference
            with torch.no_grad():
                weight_change = torch.mul(grads[j], lr)  # W = W - lr * gradient
                
            # print(f'weight_change: {weight_change}')
            # model_weights[j] = ad.sub(model_weights[j], weight_change)  # W = W - lr * gradient
            
            # update weights
            with torch.no_grad():
                model_weights[j] = torch.sub(model_weights[j], weight_change)  # W = W - lr * gradient
                
            # print(f'model_weights[j] after: {model_weights[j]}')
            # clear_cache()
            
        # Accumulate the loss
        # Add batch loss to total loss
        with torch.no_grad():
            current_batch_loss = torch.sum(loss, (0,))
            
        # print(f'current_batch: {i}')
        # print(f'current_batch_loss: {i} => {current_batch_loss}')
        
        total_loss += current_batch_loss

    # Compute the average loss
    print(f'total loss: {total_loss}')
    average_loss = (total_loss/ num_examples)
    print('Avg_loss:', average_loss)

    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 20
    lr = 0.02

    # Define the forward graph.
    # Input
    X = ad.Variable(name="X")
    
    # Weights
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    
    model_weights = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    # The output of the forward pass
    y_predict: ad.Node = transformer(X, model_weights, model_dim, seq_length, eps, batch_size, num_classes)

    y_groundtruth = ad.Variable(name="y")
    
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # Create the evaluator.
    # Define the gradient nodes here
    grads: List[ad.Node] = ad.gradients(loss, model_weights)
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().astype(np.float32).reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    # X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().astype(np.float32).reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    # X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_train, y_test = torch.tensor(y_train, dtype=torch.int8, device=device), torch.tensor(y_test, dtype=torch.int8, device=device)

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    
    # force to be float32 fo faster performance
    weight_values = [
        np.random.uniform(-stdv, stdv, (input_dim, model_dim)).astype(np.float32),  # W_Q
        np.random.uniform(-stdv, stdv, (input_dim, model_dim)).astype(np.float32),  # W_K
        np.random.uniform(-stdv, stdv, (input_dim, model_dim)).astype(np.float32),  # W_V
        np.random.uniform(-stdv, stdv, (model_dim, model_dim)).astype(np.float32),  # W_O
        np.random.uniform(-stdv, stdv, (model_dim, model_dim)).astype(np.float32),  # W_1
        np.random.uniform(-stdv, stdv, (model_dim, num_classes)).astype(np.float32),  # W_2
        np.random.uniform(-stdv, stdv, (model_dim,)).astype(np.float32),  # b_1
        np.random.uniform(-stdv, stdv, (num_classes,)).astype(np.float32)  # b_2
    ]
    model_weights = [torch.tensor(w, dtype=torch.float32, device=device, requires_grad=True) for w in weight_values]

    def f_run_model(X_val, y_val, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                # Fill in the mapping from ad.variable to tensor
                X: X_val,
                y_groundtruth: y_val,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        all_logits = []
        
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            
            logits = test_evaluator.run({
                # Fill in the mapping from variable to tensor
                X: X_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            })
            # append logits of this batch - converts logits from tensor to numpy array first
            all_logits.append(logits[0].detach().numpy())
            
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    # clear_cache(print_stats = True)
    
    for epoch in range(num_epochs):
        print(f'EPOCH: {epoch}')
        X_train, y_train = shuffle(X_train, y_train)
        
        # clear_cache(print_stats = True)
        
        print(f'TRAIN MODEL')
        model_weights, loss_val = sgd_epoch(f_run_model, X_train, y_train, model_weights, batch_size, lr)
        
        clear_cache(print_stats = True)
        
        print(f'TEST MODEL')
        # Evaluate on test set
        predictions = f_eval_model(X_test, model_weights)
        test_accuracy = np.mean(predictions == y_test.numpy())
        print(f"Epoch {epoch}: Test Accuracy = {test_accuracy:.4f}, Loss = {loss_val:.4f}\n")
        
        # clear_cache(print_stats = True)

    # Final evaluation
    final_test_accuracy = np.mean(f_eval_model(X_test, model_weights) == y_test.numpy())
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    return final_test_accuracy
    
def clear_cache(print_stats = False):
    if print_stats:
        print(f"Memory usage (before clearing cache): {psutil.virtual_memory().percent}%")
        
    gc.collect()
    # torch.mps.empty_cache()
    # torch.mps.synchronize()  # Ensures all operations finish before clearing
    
    if print_stats:
        print(f"Memory usage (after clearing cache): {psutil.virtual_memory().percent}%")
        
if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
