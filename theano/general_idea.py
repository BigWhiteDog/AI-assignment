#coding=utf8

import theano
import theano.tensor as T
from theano import function, pp, printing, scan, shared

# 1. define variables
x_symb = T.scalar('x', dtype='int32')
y_symb = T.scalar('y', dtype='int32')

# 2. define opeartions among variables
z_symb = x_symb + y_symb

# 3. define a function 
f_add = function(inputs=[x_symb, y_symb], outputs=z_symb)

# 4. use that function
x_val = 1
y_val = 2
z_val = f_add(x_val, y_val)
print z_val

