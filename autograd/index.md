# Autograd

## Background
Autograd is the basic for grad-based modelling and optimizing methods. The (online lecture)[http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/] clearly explains the autograd techniques. The basic math theory is the differentation chain.



## Augograd's ingredients
1. Tracking the composition of primitive functions;
2. Defining a vector-Jacobian product (VJP) operator for each primitive;
3. Composing VJPs backward

### Primitive function
primitive is a wrapper for each numpy function. It has three main component: unbox, function, box

