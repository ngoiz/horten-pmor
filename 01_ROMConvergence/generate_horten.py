import sharpy.utils.algebra as algebra
from cases.hangar.richards_wing import Baseline
import sharpy.sharpy_main
import numpy as np
import configobj


def run_rom_convergence(case_name, output_folder='./output/', **kwargs):
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 11)
    Msf = kwargs.get('Msf', 10)

    trim = kwargs.get('trim', True)

    rho_fact = 1.
    track_body = True
    payload = 0
    u_inf = kwargs.get('u_inf', 30)

    use_euler = True
    if use_euler:
        orient = 'euler'
    else:
        orient = 'quat'

    case_name += 'M%gN%gMsf%g_u%g' % (M, N, Msf, u_inf)

    # M4N11Msf5
    alpha_deg = 3.974
    cs_deflection = 0.3582
    thrust = 4.8062

    # M8N11Msf5
    # alpha_deg = 4.5162
    # cs_deflection = 0.2373
    # thrust = 5.5129

    # ROM settings
    rom_settings = dict()
    rom_settings['algorithm'] = 'mimo_rational_arnoldi'
    rom_settings['r'] = 10
    rom_settings['frequency'] = np.array([0], dtype=float)
    rom_settings['single_side'] = 'observability'

    case_name += 'rom_%g_%s' % (rom_settings['r'], rom_settings['single_side'][:3])

    ws = Baseline(M=M,
                  N=N,
                  Mstarfactor=Msf,
                  u_inf=u_inf,
                  rho=1.02,
                  alpha_deg=alpha_deg,  # 7.7563783342984385,
                  roll_deg=0,
                  cs_deflection_deg=cs_deflection,  # -6.733360628875144,
                  thrust=thrust,  # 10.140622253017584,
                  physical_time=20,
                  case_name=case_name,
                  case_name_format=2)

    ws.set_properties()
    ws.initialise()
    ws.clean_test_files()

    # ws.update_mass_stiffness(sigma=1., sigma_mass=1.5)
    ws.update_mass_stiffness(sigma=.5, sigma_mass=1.0, payload=payload)
    ws.update_fem_prop()
    ws.generate_fem_file()
    ws.update_aero_properties()
    ws.generate_aero_file()

    flow = ['BeamLoader',
            'AerogridLoader',
            'StaticTrim',
            'BeamPlot',
            'AerogridPlot',
            'AeroForcesCalculator',
            'DynamicCoupled',
            'Modal',
            'LinearAssembler',
            # 'LinDynamicSim',
            # 'SaveData',
            'AsymptoticStability',
            'FrequencyResponse',
            # 'StabilityDerivatives',
            # 'LinDynamicSim',
            # 'PickleData',
            'SaveParametricCase'
            ]

    if not trim:
        flow[2] = 'StaticCoupled'

    settings = dict()
    settings['SHARPy'] = {'case': ws.case_name,
                          'route': ws.case_route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': './output/' + ws.case_name + '/',
                          'log_file': ws.case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'off',
                              'orientation': algebra.euler2quat(np.array([ws.roll,
                                                                          ws.alpha,
                                                                          ws.beta]))}

    if ws.horseshoe is True:
        settings['AerogridLoader'] = {'unsteady': 'off',
                                      'aligned_grid': 'on',
                                      'mstar': 1,
                                      'freestream_dir': ['1', '0', '0'],
                                      'control_surface_deflection': ['']}
    else:
        settings['AerogridLoader'] = {'unsteady': 'off',
                                      'aligned_grid': 'on',
                                      'mstar': int(ws.M * ws.Mstarfactor),
                                      'freestream_dir': ['1', '0', '0'],
                                      'control_surface_deflection': ['']}

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': {'print_info': 'off',
                                                                'max_iterations': 200,
                                                                'num_load_steps': 1,
                                                                'delta_curved': 1e-5,
                                                                'min_delta': ws.tolerance,
                                                                'gravity_on': 'on',
                                                                'gravity': 9.81},
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': {'print_info': 'on',
                                                          'horseshoe': ws.horseshoe,
                                                          'num_cores': 4,
                                                          'n_rollup': int(0),
                                                          'rollup_dt': ws.dt,
                                                          'rollup_aic_refresh': 1,
                                                          'rollup_tolerance': 1e-4,
                                                          'velocity_field_generator': 'SteadyVelocityField',
                                                          'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                   'u_inf_direction': [1., 0, 0]},
                                                          'rho': ws.rho},
                                 'max_iter': 200,
                                 'n_load_steps': 1,
                                 'tolerance': ws.tolerance,
                                 'relaxation_factor': 0.2}

    settings['StaticTrim'] = {'solver': 'StaticCoupled',
                              'solver_settings': settings['StaticCoupled'],
                              'thrust_nodes': ws.thrust_nodes,
                              'initial_alpha': ws.alpha,
                              'initial_deflection': ws.cs_deflection,
                              'initial_thrust': ws.thrust,
                              'max_iter': 200,
                              'fz_tolerance': 1e-2,
                              'fx_tolerance': 1e-2,
                              'm_tolerance': 1e-2,
                              'save_info': 'on'}

    settings['AerogridPlot'] = {'folder': './output/',
                                'include_rbm': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': ws.u_inf
                                }
    settings['AeroForcesCalculator'] = {'folder': './output/',
                                        'write_text_file': 'off',
                                        'text_file_name': ws.case_name + '_aeroforces.csv',
                                        'screen_output': 'on',
                                        'unsteady': 'off',
                                        'coefficients': True,
                                        'q_ref': 0.5 * ws.rho * ws.u_inf ** 2,
                                        'S_ref': 12.809,
                                        }

    settings['BeamPlot'] = {'folder': './output/',
                            'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_FoR': 'on'}

    struct_solver_settings = {'print_info': 'off',
                              'initial_velocity_direction': [-1., 0., 0.],
                              'max_iterations': 950,
                              'delta_curved': 1e-6,
                              'min_delta': ws.tolerance,
                              'newmark_damp': 5e-3,
                              'gravity_on': True,
                              'gravity': 9.81,
                              'num_steps': ws.n_tstep,
                              'dt': ws.dt,
                              'initial_velocity': ws.u_inf * 1}

    step_uvlm_settings = {'print_info': 'on',
                          'horseshoe': ws.horseshoe,
                          'num_cores': 4,
                          'n_rollup': 1,
                          'convection_scheme': ws.wake_type,
                          'rollup_dt': ws.dt,
                          'rollup_aic_refresh': 1,
                          'rollup_tolerance': 1e-4,
                          'velocity_field_generator': 'SteadyVelocityField',
                          'velocity_field_input': {'u_inf': ws.u_inf * 0,
                                                   'u_inf_direction': [1., 0., 0.]},
                          'rho': ws.rho,
                          'n_time_steps': ws.n_tstep,
                          'dt': ws.dt,
                          'gamma_dot_filtering': 3}

    settings['DynamicCoupled'] = {'print_info': 'on',
                                  # 'structural_substeps': 1,
                                  # 'dynamic_relaxation': 'on',
                                  # 'clean_up_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicCoupledStep',
                                  'structural_solver_settings': struct_solver_settings,
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': step_uvlm_settings,
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': ws.fsi_tolerance,
                                  'relaxation_factor': ws.relaxation_factor,
                                  'minimum_steps': 1,
                                  'relaxation_steps': 150,
                                  'final_relaxation_factor': 0.5,
                                  'n_time_steps': 1,  # ws.n_tstep,
                                  'dt': ws.dt,
                                  'include_unsteady_force_contribution': 'off',
                                  'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'WriteVariablesTime'],
                                  'postprocessors_settings': {'BeamLoads': {'folder': './output/',
                                                                            'csv_output': 'off'},
                                                              'BeamPlot': {'folder': './output/',
                                                                           'include_rbm': 'on',
                                                                           'include_applied_forces': 'on'},
                                                              'AerogridPlot': {
                                                                  'u_inf': ws.u_inf,
                                                                  'folder': './output/',
                                                                  'include_rbm': 'on',
                                                                  'include_applied_forces': 'on',
                                                                  'minus_m_star': 0},
                                                              'WriteVariablesTime': {
                                                                  'folder': './output/',
                                                                  'cleanup_old_solution': 'on',
                                                                  'delimiter': ',',
                                                                  'FoR_variables': ['total_forces',
                                                                                    'total_gravity_forces',
                                                                                    'for_pos', 'quat'],
                                                              }}}

    settings['Modal'] = {'print_info': True,
                         'use_undamped_modes': True,
                         'NumLambda': 30,
                         'rigid_body_modes': True,
                         'write_modes_vtk': 'on',
                         'print_matrices': 'on',
                         'write_data': 'on',
                         'continuous_eigenvalues': 'off',
                         'dt': ws.dt,
                         'plot_eigenvalues': False,
                         'rigid_modes_cg': False}

    settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                   'linear_system_settings': {
                                       'beam_settings': {'modal_projection': 'on',
                                                         'inout_coords': 'modes',
                                                         'discrete_time': True,
                                                         'newmark_damp': 0.5e-2,
                                                         'discr_method': 'newmark',
                                                         'dt': ws.dt,
                                                         'proj_modes': 'undamped',
                                                         'use_euler': use_euler,
                                                         'num_modes': 20,
                                                         'print_info': 'on',
                                                         'gravity': 'on',
                                                         'remove_dofs': []},
                                       'aero_settings': {'dt': ws.dt,
                                                         # 'ScalingDict': {'length': ws.c_root,
                                                         #                 'speed': ws.u_inf,
                                                         #                 'density': ws.rho},
                                                         'integr_order': 2,
                                                         'density': ws.rho * rho_fact,
                                                         'remove_predictor': False,
                                                         'use_sparse': False,
                                                         'rigid_body_motion': True,
                                                         'use_euler': use_euler,
                                                         'remove_inputs': ['u_gust'],
                                                         'rom_method': ['Krylov'],
                                                         'rom_method_settings': {'Krylov': rom_settings}},
                                       'rigid_body_motion': True,
                                       'track_body': track_body,
                                       'use_euler': use_euler,
                                       'linearisation_tstep': -1
                                   }}

    settings['AsymptoticStability'] = {
        'print_info': 'on',
        'modes_to_plot': [],
        # 'velocity_analysis': [27, 29, 3],
        'display_root_locus': 'off',
        'frequency_cutoff': 0,
        'export_eigenvalues': 'on',
        'num_evals': 10000,
        'folder': './output/'}

    settings['FrequencyResponse'] = {'print_info': 'on',
                                     'compute_fom': 'on',
                                     'frequency_bounds': [0.1, 100]}

    settings['LinDynamicSim'] = {'dt': ws.dt,
                                 'n_tsteps': ws.n_tstep,
                                 'sys_id': 'LinearAeroelastic',
                                 # 'reference_velocity': ws.u_inf,
                                 'write_dat': ['x', 'y', 't', 'u'],
                                 # 'write_dat': 'on',
                                 'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                 'postprocessors_settings': {'AerogridPlot': {
                                     'u_inf': ws.u_inf,
                                     'folder': './output',
                                     'include_rbm': 'on',
                                     'include_applied_forces': 'on',
                                     'minus_m_star': 0},
                                     'BeamPlot': {'folder': './output/',
                                                  'include_rbm': 'on',
                                                  'include_applied_forces': 'on'}}}

    settings['StabilityDerivatives'] = {'u_inf': ws.u_inf,
                                        'S_ref': 12.809,
                                        'b_ref': ws.span,
                                        'c_ref': 0.719}

    settings['SaveData'] = {'folder': './output',
                            'save_aero': 'off',
                            'save_struct': 'off',
                            'save_linear': 'on',
                            'save_linear_uvlm': 'on'}

    settings['PickleData'] = {'folder': './output/' + ws.case_name + '/'}

    settings['SaveParametricCase'] = {'folder': './output/' + ws.case_name + '/',
                                      'parameters': {'r': rom_settings['r']}}

    config = configobj.ConfigObj()
    file_name = ws.case_route + '/' + ws.case_name + '.solver.txt'
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()

    sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.solver.txt'])


def main():

    M = 4
    N = 11
    Msf = 10

    trim = False

    case_name = 'horten_'

    run_rom_convergence(case_name=case_name, M=M, N=N, Msf=Msf, trim=trim)


if __name__ == '__main__':
    main()
