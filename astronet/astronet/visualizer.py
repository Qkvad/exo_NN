# visualizing data passing through convolutional neural network #

import numpy as np
import matplotlib.pyplot as plt


inputs = np.load('/tmp/input.npy')
convl1b1out = np.load('/tmp/convl1b1.npy')
convl1b2out = np.load('/tmp/convl1b2.npy')
pool1 = np.load('/tmp/pool1.npy')
convl2b1out = np.load('/tmp/convl2b1.npy')
convl2b2out = np.load('/tmp/convl2b2.npy')
pool2 = np.load('/tmp/pool2.npy')
flat2 = np.load('/tmp/flat2.npy')

print('input data shape     :', inputs.shape)
print('convl1b1 output shape:', convl1b1out.shape)
print('convl1b2 output shape:', convl1b2out.shape)
print('pool1 output shape   :', pool1.shape)
print('convl2b1 output shape:', convl2b1out.shape)
print('convl2b2 output shape:', convl2b2out.shape)
print('pool2 output shape   :', pool2.shape)
print('flatened output shape:', flat2.shape)


""" plot input data """
plt.scatter(range(201),inputs)
plt.show()

""" plot output of first block in first convolutional layer 
for i in range(0,convl1b1out.shape[1]):
  plt.plot(convl1b1out[:,i])
  plt.show()"""

""" plot output of second block in first convolutional layer 
for i in range(0,convl1b2out.shape[1]):
  plt.plot(convl1b2out[:,i])
  plt.show()"""

""" plot output of max pooling 1 """
fig, axs = plt.subplots(4,4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()
for i in range(pool1.shape[1]):
  axs[i].scatter(range(pool1.shape[0]),pool1[:,i],s=0.2)
  axs[i].set_title('filter '+str(i+1))
plt.show()

""" plot output of first block in second convolutional layer 
for i in range(0,convl2b1out.shape[1]):
  plt.plot(convl2b1out[:,i])
  plt.show()"""

""" plot output of second block in second convolutional layer 
for i in range(0,convl2b2out.shape[1]):
  plt.plot(convl2b2out[:,i])
  plt.show()"""

""" plot output of max pooling 2 
for i in range(0,pool2.shape[1]):
  plt.plot(pool2[:,i])
  plt.show()"""
fig, axs = plt.subplots(4,8, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()
for i in range(pool2.shape[1]):
  axs[i].scatter(range(pool2.shape[0]),pool2[:,i])
  axs[i].set_title('filter '+str(i+1))
plt.show()

plt.scatter(range(1472),flat2)
plt.show()




