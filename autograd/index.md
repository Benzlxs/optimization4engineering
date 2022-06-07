# Autograd

## 1. Background
Stochastic gradient descent approximations have been widely used in optimizing big machine learning models, and their extremely good performance on machine learning tasks unveils the power of gradient descent. For complited optimization problem, such as complicated machine learning model, the automatical calculation of gradient (Autograd) become necessary and important. Given the loss function, autograd should be able to give the gradient.




Autograd is the basic for grad-based modelling and optimizing methods. The (online lecture)[http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/] clearly explains the autograd techniques. The basic math theory is the differentation chain.



## 2. Augograd's ingredients
1. Tracking the composition of primitive functions;
2. Defining a vector-Jacobian product (VJP) operator for each primitive;
3. Composing VJPs backward

### 2.1. Primitive function
primitive is a wrapper for each numpy function. It has three main component: unbox, function, box

Node, primitive, forward_pass

### 2.2. VJP and Composing VJPs backward
vector-Jacobian product, backward_pass, make_vjp, grad

