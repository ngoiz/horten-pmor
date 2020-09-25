import sys
sys.path.append('../sharpy-analysis-tools')
from batch.interpolation import Interpolated
import linear.stability as stability
import numpy as np
import os

case_name = 'pmor_flutter_weakMAC'
output_folder = './results'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

case = Interpolated('./output/' + case_name + '/')
case.load_bulk_cases('eigs')

vel, eigs = case.eigs('aeroelastic')
dest_file = output_folder + '/velocity_eigenvalues.txt'
np.savetxt(dest_file, np.column_stack((vel, eigs)))
print('Saved velocity/eigs array to {}'.format(dest_file))

v, damp, fn = stability.modes(vel, eigs, use_hz=True, wdmax=50 * 2 * np.pi)
np.savetxt(output_folder + 'velocity_damping_frequency.txt',
           np.column_stack((v, damp, fn)))
