import numpy as np

vmin = 30
vmax = 50
n_int = 31
param_name = 'u_inf'
filename = 'input.yaml'

vec = np.linspace(vmin, vmax, n_int)

with open(filename, 'w') as f:
    for v in vec:
        f.write('- %s: %f\n' % (param_name, v))
