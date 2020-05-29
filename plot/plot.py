import numpy as np
import matplotlib.pyplot as plt

test = np.loadtxt('Two_Grid-test.txt',delimiter=',')
N = test.shape[0]
dx = 1.0/(N-1)

x = np.arange(0,dx*(N),dx)
y = np.copy(x)
phi_ref = x[:,None]*y*(np.exp(x[:,None]-y))*((x[:,None]-1)*(y-1))
phi_ref = np.flip(phi_ref,axis=1)

#fig, axes = plt.subplots(figsize=[4,10],nrows=3,ncols=1)
fig, axes = plt.subplots(figsize=[10,4],nrows=1,ncols=3)

im1 = axes[0].imshow(test,interpolation='none')
im2 = axes[1].imshow(phi_ref,interpolation='none')
im3 = axes[2].imshow(test-phi_ref,interpolation='none',cmap=plt.get_cmap('RdYlBu'))

axes[0].set_title('numerical')
axes[1].set_title('analytical')
axes[2].set_title('numerical-analytical')

fig.colorbar(im1,ax=axes[0])
fig.colorbar(im2,ax=axes[1])
fig.colorbar(im3,ax=axes[2])
fig.tight_layout()

plt.show()
