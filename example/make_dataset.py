import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x0 = np.arange(0,100,1) #daily sampling
x0[50:] += 20 # hole in the sampling
x = np.random.normal(x0, 0.1)

T0 = 12.56
P0 = 5.46
K0 = 12.93

T1 = 10.23
P1 = 23.45
K1 = 17.93

offset = 123.45

y0 =   -K0 * np.sin( (x-T0)/P0 * 2*np.pi ) +  -K1 * np.sin( (x-T1)/P1 * 2*np.pi ) + offset

x_model = np.arange(-2.5, 122.5, 0.01)
y_model = -K0 * np.sin( (x_model-T0)/P0 * 2*np.pi ) +  -K1 * np.sin( (x_model-T1)/P1 * 2*np.pi ) + offset

e0 = np.random.normal(2.0, 0.25, size=len(y0)) # inhomogeneous errors
e0[(e0<1.25)] = 1.25 #baseline for error
e = 1.0 + np.random.normal(e0)
y = np.random.normal(y0, e)


fileout = open('fake_dataset.dat', 'w')
for xx, yy, ee in zip(x,y,e):
    fileout.write('{0:12f} {1:12f} {2:12f}\n'.format(xx,yy,ee))
fileout.close()

plt.figure(figsize=(8,8))
plt.errorbar(x,y,yerr=e, fmt='o', ms=5)
plt.plot(x_model, y_model)
plt.xlabel('time [d]')
plt.ylabel('signal')
plt.savefig('fake_plot.pdf', dpi=300)
