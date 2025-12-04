import numpy as np

filename = 'metadata/bodmas.npz'
data = np.load(filename)
X = data['X']  # all the feature vectors
y = data['y']  # labels, 0 as benign, 1 as malicious

print(X.shape, y.shape)
num = 0
for i in range(len(y)):
    if y[i] == 1:
        print(i)
        num += 1

    if num == 10:
        break
