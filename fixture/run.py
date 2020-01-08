import sys, yaml, ast, os
from pathlib import Path
import fault
import fixture.templates as templates
import fixture.real_types as real_types
import fixture.sampler as sampler
import fixture.create_testbench as create_testbench
#import fixture.linearregression as lr
from fixture import Regression

def path_relative(path_to_config, path_from_config):
    ''' Interpret path names specified in config file
    We want path names relative to the current directory (or absolute).
    But we assume relative paths in the config mean relative to the config.
    '''
    if os.path.isabs(path_from_config):
        return path_from_config
    folder = os.path.dirname(path_to_config)
    res = os.path.join(folder, path_from_config)
    return res

def edit_paths(config_dict, config_filename, params):
    for param in params:
        old = config_dict[param]
        new = path_relative(config_filename, old)
        config_dict[param] = new

def run(circuit_config_filename, test_config_filename):
    with open(circuit_config_filename) as f:
        circuit_config_dict = yaml.safe_load(f)
    with open(test_config_filename) as f:
        test_config_dict = yaml.safe_load(f)
    edit_paths(circuit_config_dict, circuit_config_filename, ['filepath'])
    _run(circuit_config_dict, test_config_dict)

def _run(circuit_config_dict, test_config_dict):
    template = getattr(templates, circuit_config_dict['template'])

    # generate IO
    io = []
    pins = circuit_config_dict['pin']
    for name, p in pins.items():
        dt = getattr(real_types, p['datatype'])
        value = ast.literal_eval(str(p.get('value', None)))
        dt = dt(value)
        direction = getattr(real_types, p['direction'])
        dt = direction(dt)
        if 'width' in p:
            dt = real_types.Array(p['width'], dt)
        io += [name, dt]

    class UserCircuit(template):
        name = circuit_config_dict['name']
        IO = io
        extras = circuit_config_dict

        def mapping(self):
            for name, p in pins.items():
                if 'template_pin' in p:
                    setattr(self, p['template_pin'], getattr(self, name))
    vectors = sampler.Sampler.get_samples_for_circuit(UserCircuit, 50)

    tester = fault.Tester(UserCircuit)
    testbench = create_testbench.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()

    approved_simulator_args = ['ic', 'vsup']
    simulator_dict = {k:v for k,v in test_config_dict.items() if k in approved_simulator_args}
    print(f'Running sim, {len(vectors[0])} test vectors')
    tester.compile_and_run(test_config_dict['target'],
        simulator=test_config_dict['simulator'],
        model_paths = [Path(circuit_config_dict['filepath']).resolve()],
        clock_step_delay=0,
        **simulator_dict
    )
    
    print('Analyzing results')
    results = testbench.get_results()

    results_mode_0 = results[0]
    reg = Regression(UserCircuit, results_mode_0)


if __name__ == '__main__':
    args = sys.argv
    circuit_config_filename = args[1]
    test_config_filename = args[2]
    run(circuit_config_filename, test_config_filename)
