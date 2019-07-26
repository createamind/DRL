
import numpy as np
from collections import deque

frames = deque([], maxlen=3)

frames.append(2)
frames.append([False,[2,3]])
frames.append([True,[2,2]])
frames.append([True,[2,2]])
# frames.extend([4,[2,2]])

for _i in range(10):
    frames.append((np.zeros((5,), dtype=np.float32), False))
print(frames)

a = np.stack(frames, axis=0)
print(a[:,0])
print(a[:,1])



arr = np.array([['one', [1, 2, 3]],['two', [4, 5, 6]]], dtype=np.object)
arr0 = np.array([['one', [1, 2, 3]],['two', [4, 5, 6]]])
arr1 = np.array(list(arr[:, 1]), dtype=np.float)

print(arr1)

arr2 = np.array([np.array([2,2,2]),np.array([2,2,2]),np.array([2,2,2]),np.array([2,2,2])])

print(arr2)