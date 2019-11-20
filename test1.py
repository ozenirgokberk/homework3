import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_dataset_by_dimension_cnt(n):
    X = []
    y = []
    t = []
    a = []
    for j in range(1, (n**2)+2, 1):
        if len(X) == n:
            t.append(X)
            X=[]
        exec('var_%d = j+np.random.rand()*j' % j)
        exec('X=X.append(var_%d)' % j)
    for i in range(n):
        y.append([i + np.random.rand() + i])
        a.append(0)
    coef = np.array(a)
    return np.array(t), np.array(y), coef

def generate_dataset(n):
    X = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i / 2 + np.random.rand() * n
        X.append([1, x1, x2])
        y.append([random_x1 * x1 + random_x2 * x2 + 1])
    return np.array(X), np.array(y)

def mean_square_error(coef, x, y):
    return np.mean((np.dot(x, coef) - y) ** 2) / 2

def gradients(coef, x, y):
    return np.mean(x.transpose() * (np.dot(x, coef) - y), axis=1)

def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8):
    prev_error = 0
    m_coef = np.zeros(coef.shape)
    v_coef = np.zeros(coef.shape)
    moment_m_coef = np.zeros(coef.shape)
    moment_v_coef = np.zeros(coef.shape)
    t = 0

    while True:
        error = mean_square_error(coef, x, y)
        if abs(error - prev_error) <= epsilon:
            break
        prev_error = error
        try:
            grad = gradients(coef, x, y)
        except Exception as e:
            print(e)
        t += 1
        m_coef = b1 * m_coef + (1 - b1) * grad
        v_coef = b2 * v_coef + (1 - b2) * grad ** 2
        moment_m_coef = m_coef / (1 - b1 ** t)
        moment_v_coef = v_coef / (1 - b2 ** t)

        delta = ((lr / moment_v_coef ** 0.5 + 1e-8) *
                 (b1 * moment_m_coef + (1 - b1) * grad / (1 - b1 ** t)))

        coef = np.subtract(coef, delta)
    return coef


n = 10
x, y, coef = generate_dataset_by_dimension_cnt(n)

#x,y = generate_dataset(3)
#coef = np.array([0, 0, 0])
mpl.rcParams['legend.fontsize'] = 16

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x[:, 1], x[:, 2], y, label='y', s=n)
ax.legend()
ax.view_init(45, 0)
plt.show()


c = multilinear_regression(coef, x, y, 1e-1)
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x[:, 1], x[:, 2], y, label='y',
           s=5, color="dodgerblue")

ax.scatter(x[:, 1], x[:, 2], c[0] + c[1] * x[:, 1] + c[2] * x[:, 2],
           label='regression', s=5, color="orange")

ax.view_init(45, 0)
ax.legend()
plt.show()
