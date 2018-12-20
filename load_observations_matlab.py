import numpy as np
import scipy.io
% IMPORT YOUR OWN LIBRARIES TO RUN YOUR SIMULATED ANNEALING

var = scipy.io.loadmat('observations.mat')
Y = var['Y']
Y = Y.reshape(Y.size)
W = var['W']
m = int(var['m'])
n = int(var['n'])

print(np.shape(Y))
print(np.shape(W))
print(m)
print(n)

% RUN YOUR SIMULATED ANNEALING
% x_hat IS THE ESTIMATE RETURNED BY YOUR ALGORITHM

scipy.io.savemat('YOUR_TEAM_NAME', {'x_estimate':x_hat}, appendmat=True, format='5', oned_as='column')
