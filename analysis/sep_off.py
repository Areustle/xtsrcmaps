import numpy as np

xmin = 1e-4
xmax = 70.0
xmon = 1e4
N = 400
d = np.log(xmax * xmon) / (N - 1)
sep = np.zeros(N + 1)

# xx.clear();
# xx.reserve(npts+1);
# xx.push_back(0);
# double xstep = log(xmax/xmin)/(npts - 1.);
# for (unsigned int i = 0; i < npts; i++) {
#    xx.push_back(xmin*exp(i*xstep));
# }


def f(i):
    return xmin * np.exp(i * d)


y = np.arange(0, N, dtype=np.int64)
print(y)

# sep = f(y)
for i in y:
    sep[i + 1] = f(i)


print(sep)


def g(s):
    return np.clip(1 + np.log(s / xmin) / d, 0.0, N - 1)


print(g(sep))

print(sep[:10], sep[390:])
print(g(sep)[:10], g(sep)[390:])
print(g(0.0))
print(g(1e-7))
print(g(0.9999e-4), g(1.0000e-4), g(1.00001e-4))
