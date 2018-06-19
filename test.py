import theano.tensor as T
from theano import function
import numpy as np

x = T.matrix('x')
y = T.matrix('x')
linmax = function([x,y], [T.maximum(x,y)])


a = np.array([[2,1],[-2,3]]).astype(np.float32)
b = np.array([[-2,3],[-1,5]]).astype(np.float32)

print a
print b
print linmax(a,b)
