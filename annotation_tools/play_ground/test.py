import numpy as np 
a = np.array([[1,-2,3,], [2,-9,-1]])
print(a)
m = np.argwhere(a < 0)
print(m)