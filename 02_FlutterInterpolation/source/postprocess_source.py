import sys
sys.path.append('../../sharpy-analysis-tools')
from batch.sets import Actual
import linear.stability as stability
import numpy as np
import os

set_name = 'output_source_30_50_11_modesign_scaled_uvlm'
output_folder = './results/'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


case = Actual('./' + set_name + '/*')
case.load_bulk_cases('eigs', eigs_legacy=False)

vel, eigs = case.eigs('aeroelastic')
dest_file = output_folder + '/velocity_eigenvalues.txt'
np.savetxt(dest_file, np.column_stack((vel, eigs)))
print('Saved velocity/eigs array to {}'.format(dest_file))

v, damp, fn = stability.modes(vel, eigs, use_hz=True, wdmax=50 * 2 * np.pi)
np.savetxt(output_folder + 'velocity_damping_frequency.txt',
           np.column_stack((v, damp, fn)))
