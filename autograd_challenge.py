import numpy as np

class Tensor:
    '''
    Class that imitates PyTorch tensors
    '''
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.grad_fn = None # Function used to compute gradient
        self.requires_grad = requires_grad
        self.children = [] # Holds the next tensor(s) in backpropagation

    def __str__(self):
        return f"tensor({self.data})"
    
    def backprop(self, grad):
        '''
        Backpropagation gradient calculations
        Iterates through the computation graph recursively
        '''
        if not self.requires_grad:
            print("No gradient required for input")
            return

        # Set the gradient for the current layer/tensor
        self.grad = grad

        if self.grad_fn:
            grads = self.grad_fn(grad) # Compute the gradient
            for child, g in zip(self.children, grads): # For each child tensor, backpropagate the gradient
                child.backprop(g)

class SimpleNN:
    def __init__(self, shape=[5, 10, 8, 5, 1]):
        '''
        Initialise the layers of the network using the reference as an example shape
        Consists of 5 layers which require 4 sets of weights and biases
        '''
        self.w1, self.b1 = init_layer(shape[0], shape[1])
        self.w2, self.b2 = init_layer(shape[1], shape[2])
        self.w3, self.b3 = init_layer(shape[2], shape[3])
        self.w4, self.b4 = init_layer(shape[3], shape[4])
        
    def forward(self, x):
        '''
        Forward pass through network

        Params:
        x = inputs to network
        '''
        v1 = add(matmul(x, self.w1), self.b1)
        a1 = relu(v1)
        v2 = add(matmul(a1, self.w2), self.b2)
        a2 = relu(v2)
        v3 = add(matmul(a2, self.w3), self.b3)
        a3 = relu(v3)
        self.out = add(matmul(a3, self.w4), self.b4)

    def backward(self, y_true, lr=0.01):
        '''
        Backward pass through network with weight/bias updates

        Params:
        y_true = target outputs for network
        lr = learning rate
        '''

        # Compute the MSE loss
        self.loss, diff = mse(y_true, self.out)

        # Begin backpropagation recursion
        self.loss.backprop(diff)

        # Update weights and biases
        for i in [self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4]:
            i.data -= lr * i.grad

def init_layer(i, o):
    '''
    Initialise a layer's weights and biases
    
    Params:
    i = number of input nodes to layer
    o = number of output nodes from layer
    
    Returns:
    w = weights matrix
    b = biases matrix
    '''

    w = np.random.uniform(low=-1, high=1, size=(i,o))
    b = np.random.uniform(low=-1, high=1, size=(1,o))
    return Tensor(w, True), Tensor(b, True)

def relu(t):
    '''
    Define a gradient function for ReLU
    Apply ReLU activation function to each element in a tensors data
    Create the resulting tensor, while assigning certain attributes

    Params:
    t = input tensor
    '''
    def grad_fn(grad):
        # Only set the gradient if it is required
        t_grad = grad * (t.data > 0) if t.requires_grad else None
        return (t_grad,)
    
    out = Tensor(np.maximum(0, t.data), t.requires_grad)
    out.grad_fn = grad_fn
    out.children = [t]
    return out

def add(t1, t2):
    '''
    Define a gradient function for adding tensors
    Add both tensor's data together
    Create the resulting tensor, while assigning certain attributes

    Params:
    t1, t2 = input tensors
    '''
    def grad_fn(grad):
        t1_grad = grad if t1.requires_grad else None
        t2_grad = grad if t2.requires_grad else None
        return (t1_grad, t2_grad)

    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data + t2.data, requires_grad)
    out.grad_fn = grad_fn
    out.children = [t1, t2]
    return out

def matmul(t1, t2):
    '''
    Define a gradient function for the matrix multiplication of tensors
    Multiple the tensor data matrices together
    Create the resulting tensor, while assigning certain attributes

    Params:
    t1, t2 = input tensors
    '''
    def grad_fn(grad):
        t1_grad = np.matmul(grad, t2.data.T) if t1.requires_grad else None
        t2_grad = np.matmul(t1.data.T, grad) if t2.requires_grad else None
        return (t1_grad, t2_grad)
    
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(np.matmul(t1.data, t2.data), requires_grad)
    out.grad_fn = grad_fn
    out.children = [t1, t2]
    return out

def mse(y_true, y_pred):
    '''
    Define a gradient function for MSE loss
    Find the MSE loss of the predicted output
    Create the resulting tensor, while assigning certain attributes

    Params:
    y_true = target
    y_pred = predicted output of forward pass

    Returns:
    loss = loss tensor, used to start backpropagation
    diff = difference between y_pred and y_true
    '''
    def grad_fn(diff):
        return 2 * diff / diff.size

    diff = y_pred.data - y_true.data # Save this difference for MSE grad_fn later
    loss = Tensor(np.mean(diff * diff), True)
    loss.children = [y_pred]
    loss.grad_fn = grad_fn
    return loss, np.array([diff])

# Instantiate the model
model = SimpleNN()

# Hardcoded input and target
inputs = Tensor(np.array([[0.5, -0.2, 0.1, 0.7, -0.3]]))
target = Tensor(np.array([[1.0]]))

# Training for one epoch
model.forward(inputs)
model.backward(target)

# Print the gradient of the first layer
print(f"tensor({model.w1.grad.T})")

# Print the output and target for verification
print("Deep model output:", model.out)
print("Target:", target)