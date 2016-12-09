# coding=utf8

import theano
import theano.tensor as T
from theano import function, pp, printing, scan, shared
import numpy as np


'''Define a function'''
# 1. theano function
x, y, a, b = T.fscalars('x', 'y', 'a', 'b')
a = x + y
b = x - y
f_add = theano.function(inputs=[x, y], outputs=a)   
f_add_minus = theano.function(inputs=[x, y], outputs=[a, b])   

x1 = 10.
y1 = 1.
a1 = f_add(x1, y1)
a2, b2 = f_add_minus(x1, y1)
print a1, a2, b2
# 2. python function
def f_add(x, y):
    a = x + y
    return a

def f_add_minus(x, y):
    a = x + y
    b = x - y
    return a, b

x1 = 10.
y1 = 1.
a1 = f_add(x1, y1)
a2, b2 = f_add_minus(x1, y1)
print a1, a2, b2



''' Derivatives '''
# Derivative
x = T.scalar('x', dtype='float32')
y = x ** 2
gy = T.grad(y, x)
f = theano.function([x], gy)
print f(4)

# partial derivative
x = T.scalar('x', dtype='float32') 
a = T.scalar('a', dtype='float32') 
y = x * a
gy = T.grad(y, [x, a])
f = theano.function([x, a], gy)
print f(4, 3)


''' Debugging tricks '''
x = T.scalar('x',dtype='float32')
y = x ** 2
gy = T.grad(y, x)
f = function([x], gy)

# using pp
pp(gy)   # before optimization
pp(f.maker.fgraph.outputs[0])  # after optimization

# using printing.debugprinting
printing.debugprint(gy)  # before optimization
printing.debugpring(f.maker.fgraph.outputs[0])  # after optimization

# printing.Print
# recall that functions defined before do not print internal variables
x = T.scalar('x',dtype='float32')
xp2 = x + 2
xp2_printed = printing.Print('this is xp2:')(xp2)
xp2m2 = xp2_printed * 2
f = function([x], xp2m2)
f(20)

# try to make this work
x = T.scalar('x',dtype='float32')
y = x ** 2
yprint = printing.Print('this is y:')(y)
gy = T.grad(yprint, x)
f = function([x], gy)
# alternatively:
# p = printing.Print('y')
# yprint = p(y)


''' shared variable '''
state = shared(0.)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print state.get_value()
z = accumulator(10)
print z
print state.get_value()


''' Loop '''
# python for loop
import numpy
A = numpy.array([1, 2], dtype='float32')
k = 5
result = [numpy.array([1,1])]
def mul(a, b): return a*b 

for i in range(k):
    result.append(mul(result[-1], A))

print result[-1]

# theano scan
k = T.scalar("k", dtype='int32')
A = T.vector("A", dtype='float32')
def mul(a, b): return a*b

result, updates = theano.scan(
                fn=mul,
                outputs_info=T.ones_like(A),
                non_sequences=A,
                n_steps=k)

power = theano.function(
                inputs=[A,k], 
                outputs=result[-1], 
                updates=updates)

A_val = numpy.array([1, 2], dtype='float32')
k_val = 5
r = power(A_val, 5)
print r


# calculate polynomial
# that is, a0 * x^0 + a1 * x^1 + a2 * x^2 + ... + an * x^n
coefficients = theano.tensor.vector("coefficients", dtype='float32')
x = T.scalar("x", dtype='float32')

max_coefficients_supported = 10000

def cumulative_poly(coeff, power, prior_sum, x):
    return prior_sum + coeff * (x ** power)

# Generate the components of the polynomial
zero = np.asarray(0., dtype='float32')
full_range=theano.tensor.arange(max_coefficients_supported, dtype='float32')
results, updates = theano.scan(fn=cumulative_poly, 
                                outputs_info=T.as_tensor_variable(zero),
                                sequences=[coefficients, full_range],
                                non_sequences=x)

polynomial = results[-1]
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                     outputs=polynomial)

test_coeff = np.asarray([1, 0, 2], dtype=np.float32)
print(calculate_polynomial(test_coeff, 3))