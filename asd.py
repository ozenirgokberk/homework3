from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def axesDimensions(ax):
    if hasattr(ax, 'get_zlim'):
        return 3
    else:
        return 2


fig = plt.figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, projection='3d')

print ("ax1: ", axesDimensions(ax1))
print ("ax2: ", axesDimensions(ax2))