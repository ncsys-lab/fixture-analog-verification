import ast
import functools

import numpy as npy
import scipy


from fixture import TemplateMaster, PlotHelper
from fixture import template_creation_utils
from fixture import signals
import fixture
from fixture.signals import SignalOut
from fixture.signals import create_input_domain_signal
Regression = fixture.regression.Regression
import math
import matplotlib.pyplot as plt
from fault import domain_read
from fixture import ChannelUtil
from fixture.template_creation_utils import dynamic, extract_pzs, remove_repeated_timesteps

class ComparatorLatchTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'clk', 'outp', 'outn']
    required_info = {
        #'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)',
        #'max_slope': 'Approximate maximum slope of the input signal to be sampled (V/s)'
        'clks': 'Timing information for one period including clock edges and sampling and read times'
    }

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 11 # 31

        # we have to do this before getting input domains, which happens
        # in the call to super
        extras = args[3]
        signal_manager = args[2]

        # NOTE I think this block has to be before super() because that's when
        # the individual tests get copies of these signals ??
        # but must come after for self.signals to be defined?
        if 'clks' in extras:
            settle = float(extras['clks']['unit']) * float(extras['clks']['period'])
            if 'approx_settling_time' not in extras:
                extras['approx_settling_time'] = settle
            extras['cycle_time'] = settle

            clks = extras['clks']
            #clks = {k: v for k, v in clks.items() if (k != 'unit' and k != 'period')}
            for clk, v in clks.items():
                if type(v) == dict and 'max_jitter' in v:
                    x = v['max_jitter']
                    signal_manager.add(signals.SignalIn(
                        (-x, x),
                        'analog',
                        True,
                        False,
                        None,
                        None,
                        clk+'_jitter',
                        True
                    ))


        super().__init__(*args, **kwargs)


        # NOTE this must be after super() because it needs ports to be defined

        #if not hasattr(self.ports.clk, '__getitem__'):
        #    #self.ports.mapping['clk'] = [self.ports.mapping['clk']]
        #    #self.ports.mapping['out'] = [self.ports.mapping['out']]
        #    pass
        #else:
        #    assert len(self.ports.out) == len(self.ports.clk), \
        #        'Must have one clock for each output!'


    def read_value(self, tester, port, wait):
        tester.delay(wait)
        s = self.signals.from_circuit_pin(port)
        return tester.get_value(s)

    def interpret_value(self, read):
        return read.value

    def get_clock_offset_domain(self):
        return []

    def schedule_clk(self, tester, output, num_periods=1, pos=0.5, jitters={}):
        # play num_periods periods, each time with "output" coming after
        # period_len*pos
        assert 'clks' in self.extras

        clks = self.extras['clks']
        if hasattr(self.signals, 'ignore'):
            for s_ignore in self.signals.ignore:
                if s_ignore is None:
                    continue
                if s_ignore.spice_name not in clks:
                    tester.poke(s_ignore.spice_pin, 0)

        if isinstance(self.signals.clk, list):
            clk_list = self.signals.clk
            num_samplers = len(clk_list)
            #desired_sampler = clk_list.index(main)
        else:
            #main = self.get_name_circuit(port)
            num_samplers = 1
            #desired_sampler = 0
        unit = float(clks['unit'])
        period = clks['period']
        #clks = {self.signals.from_circuit_name(k): v for k,v in clks.items()
        #        if (k!='unit' and k!='period')}
        clks_new = {}
        outs = {}
        for k, v in clks.items():
            try:
                if isinstance(k, SignalOut):
                    x = k
                else:
                    x = self.signals.from_circuit_name(k)
                if x in self.signals.clk:
                    clks_new[x] = v
                elif x in self.signals.out:
                    outs[x] = v
                elif x in self.signals.ignore:
                    # if it's in clks, we should poke it
                    clks_new[x] = v
                else:
                    assert False, f'clk dict port {x} not clk or out or ignore'
            except KeyError:
                continue
        clks = clks_new


        # main_period_start_time used to be rising edge of main clock
        # we were trying to put the falling edge in the middle of the period
        # Now we want to sampling time in the middle of the period
        #assert main in clks, f'"clks" spec missing {main}'
        #main_period_start_time = [t for t,v in clks[main].items() if v==1][0]
        assert output in outs
        main_dict_rev = {v: k for k, v in outs[output].items()}
        sample_time = main_dict_rev['sample']
        time_sample_to_read = (main_dict_rev['read'] - main_dict_rev['sample']) * unit
        time_sample_to_read %= (period * unit)

        # shift the user-given period such that main_period_start_time is position
        # rising edge just happened

        for i in range(num_samplers):
            #offset = ((i - desired_sampler + num_samplers) % num_samplers) / num_samplers
            #period_start_time = (main_period_start_time + period * offset) % period
            # pick start time s.t. X plays after waiting pos*period
            period_start_time = (sample_time - pos * period + period) % period

            clks_transform = {}

            for clk in clks:
                temp = []
                name = clk[0].spice_name if isinstance(clk, list) else clk.spice_name
                jitter_name = name + '_jitter'
                jitter = jitters.get(jitter_name, 0)
                for p in range(num_periods):
                    for time, val in clks[clk].items():
                        if time == 'max_jitter':
                            # not actually a time spec,
                            # should maybe be removed from dict earleir
                            continue
                        time_transform = time + (period - period_start_time)
                        # TODO is there a reason we didn't mod here?
                        if time_transform > period:
                            time_transform -= period
                        time_transform_shift = time_transform + p * period
                        temp.append((time_transform_shift + jitter, val))
                clks_transform[clk] = sorted(temp)

            # shift that one period s.t. the falling edge of the main clk
            # happens after exactly "wait"
            # simply ignore any edges that would've been in the past
            # shift is the time in seconds from now until the period start
            shift = 0# wait - period_start_time * unit
            if shift < 0:
                print('Cannot run a full period when scheduling clk edges', i)

            for clk, edges in clks_transform.items():
                t = 0
                waits = [0]
                values = [0 if edges[0][1] else 1]
                for time, value in edges:
                    x = time * unit + shift
                    if x < 0:
                        print('Skipping edge', value, 'for', clk, i)
                        continue
                    waits.append(x - t)
                    t = x
                    values.append(value)
                #current_clk_bus = getattr(self.dut, clk)
                if not hasattr(clk, '__getitem__'):
                    current_clk = clk
                else:
                    current_clk = clk[i]
                assert waits[0] >= 0 and all(w > 0 for w in waits[1:])
                tester.poke(current_clk.spice_pin, 0, delay={
                    'type': 'future',
                    'waits': waits,
                    'values': values
                })

        # amount of time between sampling and looking at the sampled voltage
        return time_sample_to_read


    class DynamicTest(TemplateMaster.Test):

        
        num_samples = 1 #10

        def __init__(self, *args, **kwargs):
            print("STATIC INIT")
            # set parameter algebra before parent checks it
            nl_points = args[0].nonlinearity_points
            self.IS_DEBUG_MODE = True
            self.parameter_algebra = {
            #'p1': {'cm_to_p1': 'in_cm', 'const_p1': '1'},
            #'p2': {'cm_to_p2': 'in_cm', 'const_p2': '1'},
            #'z1': {'cm_to_z1': 'in_cm', 'const_z1': '1'},
            'p1': {'const_p1': '1'},
            'p2': {'const_p2': '1'},
            #'z1': {'const_z1': '1'},
        }
            super().__init__(*args, **kwargs)

        def input_domain(self):
            in_cm = create_input_domain_signal('in_cm', self.extras['limits_cm'])
            vmin, vmax = self.extras['limits_diff']
            step_size = create_input_domain_signal('step_size', ((vmax-vmin)/2, vmax-vmin))
            step_pos = create_input_domain_signal('step_pos', (0, 1))
            return [in_cm, step_size, step_pos] +  self.template.get_clock_offset_domain() 

        def testbench(self, tester, values):
            wait_time = float(self.extras['approx_settling_time'])*2
            #print('Chose jitter value', values['clk_v2t_l_jitter'], 'In static test')
            period = float(self.extras['cycle_time'])
            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk
            assert isinstance(clk, signals.SignalIn)

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_time = np * period * 22
            self.debug(tester, clk.spice_pin, debug_time)
            #for p in self.ports.out:
            #    self.debug(tester, p, debug_time)

            print(self.ports)
            self.debug(tester, self.ports.inp, debug_time)
            self.debug(tester, self.ports.inn, debug_time)
            if hasattr(self.ports, 'debug'):
                self.debug(tester, self.ports.debug, debug_time)

            #self.debug(tester, self.template.dut.z_debug, debug_time)

            #self.debug(tester, self.template.dut.clk_v2t_e[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_e[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_eb[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_eb[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_gated[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_gated[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_l[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_l[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_lb[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_lb[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[1], debug_time)

            # prevents overlapping points from confusing fault
            
            clko = tester.get_value(self.ports.clk, params=
                    {'style':'block', 'duration': period + wait_time * 2}
                )

            rp = tester.get_value(self.ports.outp, params=
                    {'style':'block', 'duration': period + wait_time * 2}
                )
            
            rn = tester.get_value(self.ports.outn, params=
                    {'style':'block', 'duration': period + wait_time * 2}
                )
        
            inp = tester.get_value(self.ports.inp, params=
                    {'style':'block', 'duration': period + wait_time * 2}
                )
            
            inn = tester.get_value(self.ports.inn, params=
                    {'style':'block', 'duration': period + wait_time * 2}
                )


            tester.poke(self.ports.clk, 1)
            tester.delay(wait_time)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?

            in_cm = values['in_cm']
            vmin, vmax = self.extras['limits_diff']
            step_start = vmin + ((vmax-vmin) - values['step_size']) * values['step_pos']
            step_end = step_start + values['step_size']
            inp_start, inn_start = in_cm + step_start/2, in_cm - step_start/2
            inp_end, inn_end = in_cm + step_end/2, in_cm - step_end/2

            tester.poke(self.ports.inp, inp_start)
            tester.poke(self.ports.inn, inn_start)
            tester.delay(wait_time)

            tester.poke(self.ports.clk, 1)
            tester.poke(self.ports.inp, inp_end)
            tester.poke(self.ports.inn, inn_end)





            """
            limits = self.signals.inn.value
            print("LIMITS: {}".format(limits))
            limits = self.signals.inp.value
            num = self.template.nonlinearity_points
            results = []
            prev_val = 0
            for i in range(num):
                #dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                #tester.poke(clk.spice_pin, 1)
                tester.poke(self.ports.in_, prev_val)
                tester.delay(0.01 * period)
                tester.poke(self.ports.in_, dc)
                prev_val = dc

                settle_time = self.template.schedule_clk(tester, self.signals.out[0], 1, 0.5, values)
                tester.delay(period / 2)
                print('1: delaying', period/2)

                # delays time "wait" for things to settle before reading
                tester.delay(settle_time)
                print('2: delaying ', settle_time)
                read = self.template.read_value(tester, p, 0)

                if settle_time < period / 2:
                    tester.delay(period / 2 - settle_time)
                    print('3: delaying', period/2 - settle_time)
                results.append((dc, read))
            tester.poke(self.ports.in_, prev_val)
            tester.delay(0.01*period)


            tester.delay(2*period)
            """



            """
            print("========================rp========================")
            print(rp)
            print("========================rn========================")
            print(rn)
            """
            tester.delay(wait_time * 1.1)
            return [rp, rn, inn, inp, clko]

        def analysis(self, reads):
            
            print("========================outp========================")
            print(reads)

            outp = reads[0].value
            outn = reads[1].value
            inn = reads[2].value
            inp = reads[3].value
            clko = reads[4].value

            print("========================outp========================")
            print(outp[1])
            print("========================outn========================")
            print(outn[1])

            print("IN_ANALYSIS_STAGE")
            # haven't written good logic for if the timesteps don't match
            if len(outp[0]) != len(outn[0]) or any(outp[0] != outn[0]):
                print('interpolating to match timesteps')
                outp = remove_repeated_timesteps(*outp)
                outn = remove_repeated_timesteps(*outn)
                inn  = remove_repeated_timesteps(*inn)
                inp  = remove_repeated_timesteps(*inp)
                clko  = remove_repeated_timesteps(*clko)

                # hmm, timesteps don't match
                # reasmple n with p's timesteps? Not ideal, but good enough
                interpn = interpolate.InterpolatedUnivariateSpline(outn[0], outn[1])
                interpinp = interpolate.InterpolatedUnivariateSpline(inn[0], inn[1])
                interpinn = interpolate.InterpolatedUnivariateSpline(inp[0], inp[1])
                interpcko = interpolate.InterpolatedUnivariateSpline(clko[0], clko[1])
                resampled_outn = interpn(outp[0])
                resampled_inn  = interpinn(outp[0])
                resampled_inp  = interpinp(outp[0])
                resampled_clko = interpcko(clko[0])
                outn = outp[0], resampled_outn
                inp  = outp[0], resampled_inn
                inn  = outp[0], resampled_inp
                clko = outp[0], resampled_clko

            print("========================inp========================")
            print(inp[1])
            print("========================inn========================")
            print(inn[1])
            print("========================clk========================")
            print(clko[1])

            # we want to cut some off, but leave at least 60-15*2 ??
            CUTOFF = 0#min(max(0, len(outp[0]) - 60), 15)

            step_start_output = outp[1][0] - outn[1][0]
            outdiff = outp[0], outp[1] - outn[1] - step_start_output

            figure, axis = plt.subplots(4,1, sharex=True)
            
            axis[0].plot( outp[0] , outp[1], label = "outp" )
            axis[0].set_title("outp")
            axis[0].set_ylabel('voltage')
            axis[0].grid()
            axis[1].plot( outn[0] , outn[1], label = "outn" )
            axis[1].set_title("outn")
            axis[1].set_ylabel('voltage')
            axis[1].grid()
            axis[2].plot( inn[0], inp[1] - inn[1],   label = "inn" )
            axis[2].set_title("delta")
            axis[2].set_ylabel('voltage')
            axis[2].grid()            
            axis[3].plot( clko[0], clko[1],   label = "clk" )
            axis[3].set_title("clk")
            axis[3].set_ylabel('voltage')
            axis[3].grid()
            figure.suptitle(f'Plot for transient')

            #plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
            #plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')

            plt.show()

            # FLIP
            #outdiff = outdiff[0], -1 * outdiff[1]


            ps, zs = extract_pzs(2, 1, outdiff[0][CUTOFF:], outdiff[1][CUTOFF:])
            list(ps).sort(key=abs)
            zs.sort()
            print(ps)
            print(zs)

            return {'p1' : ps[0], 'p2': ps[1]}


        def post_regression(self, results, data):
            #return {}
            if hasattr(self, 'IS_DEBUG_MODE'):
                for param in results.keys():
                    reg = results[param]

                    y_meas = reg.model.endog
                    y_pred = reg.model.predict(reg.params)

                    plt.scatter(y_meas, y_pred)
                    plt.title(f'Plot for {param}')
                    plt.xlabel('Measured output values')
                    plt.ylabel('Predicted output values based on inputs & model')
                    #plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
                    #plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')
                    plt.grid()
                    plt.show()

            return {}

    class SweepTest(TemplateMaster.Test):

        
        num_samples = 2 #10

        def __init__(self, *args, **kwargs):
            print("STATIC INIT")
            # set parameter algebra before parent checks it
            nl_points = args[0].nonlinearity_points
            self.IS_DEBUG_MODE = True
            self.parameter_algebra = {
            #'p1': {'cm_to_p1': 'in_cm', 'const_p1': '1'},
            #'p2': {'cm_to_p2': 'in_cm', 'const_p2': '1'},
            #'z1': {'cm_to_z1': 'in_cm', 'const_z1': '1'},
            'p1': {'const_p1': '1'},
            'p2': {'const_p2': '1'},
            #'z1': {'const_z1': '1'},
        }
            super().__init__(*args, **kwargs)

        def input_domain(self):

            return []

        def testbench(self, tester, values):


            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk

            cycles = npy.linspace(self.extras['sweep_values'][0], self.extras['sweep_values'][1])

            measure_time = self.extras['sweep_dt'] * (len(cycles) + 1)

            clko = tester.get_value(self.ports.clk, params=
                    {'style':'block', 'duration': measure_time}
                )

            rp = tester.get_value(self.ports.outp, params=
                    {'style':'block', 'duration': measure_time}
                )
            
            rn = tester.get_value(self.ports.outn, params=
                    {'style':'block', 'duration': measure_time}
                )
        
            inp = tester.get_value(self.ports.inp, params=
                    {'style':'block', 'duration': measure_time}
                )
            
            inn = tester.get_value(self.ports.inn, params=
                    {'style':'block', 'duration': measure_time}
                )

            tester.poke(self.ports.clk, 0)
            tester.poke(self.ports.inp, 0)
            tester.poke(self.ports.inn, self.extras['competing_v'])
            tester.delay(10 * self.extras['sweep_dt'])


            tester.poke(self.ports.clk, 1)
            tester.poke(self.ports.inn, self.extras['competing_v'])
            tester.delay(self.extras['sweep_dt'])

            time = 0
            period = self.extras['clks']['period']



            for i, v in enumerate(cycles):
                tester.poke(self.ports.inp, v)
                #if i > (len(cycles) // 2):
                #    tester.poke(self.ports.clk, 0)
                #if i > (len(cycles) // 2) + 10:
                #    tester.poke(self.ports.clk, 1)


                tester.delay(self.extras['sweep_dt'])




            """
            limits = self.signals.inn.value
            print("LIMITS: {}".format(limits))
            limits = self.signals.inp.value
            num = self.template.nonlinearity_points
            results = []
            prev_val = 0
            for i in range(num):
                #dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                #tester.poke(clk.spice_pin, 1)
                tester.poke(self.ports.in_, prev_val)
                tester.delay(0.01 * period)
                tester.poke(self.ports.in_, dc)
                prev_val = dc

                settle_time = self.template.schedule_clk(tester, self.signals.out[0], 1, 0.5, values)
                tester.delay(period / 2)
                print('1: delaying', period/2)

                # delays time "wait" for things to settle before reading
                tester.delay(settle_time)
                print('2: delaying ', settle_time)
                read = self.template.read_value(tester, p, 0)

                if settle_time < period / 2:
                    tester.delay(period / 2 - settle_time)
                    print('3: delaying', period/2 - settle_time)
                results.append((dc, read))
            tester.poke(self.ports.in_, prev_val)
            tester.delay(0.01*period)


            tester.delay(2*period)
            """



            """
            print("========================rp========================")
            print(rp)
            print("========================rn========================")
            print(rn)
            """

            return [rp, rn, inn, inp, clko]

        def analysis(self, reads):
            
            print("========================outp========================")
            print(reads)

            outp = reads[0].value
            outn = reads[1].value
            inn = reads[2].value
            inp = reads[3].value
            clko = reads[4].value

            print("========================outp========================")
            print(outp[1])
            print("========================outn========================")
            print(outn[1])

            print("IN_ANALYSIS_STAGE")
            # haven't written good logic for if the timesteps don't match
            if len(outp[0]) != len(outn[0]) or any(outp[0] != outn[0]):
                print('interpolating to match timesteps')
                outp = remove_repeated_timesteps(*outp)
                outn = remove_repeated_timesteps(*outn)
                inn  = remove_repeated_timesteps(*inn)
                inp  = remove_repeated_timesteps(*inp)
                clko  = remove_repeated_timesteps(*clko)

                # hmm, timesteps don't match
                # reasmple n with p's timesteps? Not ideal, but good enough
                interpn = interpolate.InterpolatedUnivariateSpline(outn[0], outn[1])
                interpinp = interpolate.InterpolatedUnivariateSpline(inn[0], inn[1])
                interpinn = interpolate.InterpolatedUnivariateSpline(inp[0], inp[1])
                interpcko = interpolate.InterpolatedUnivariateSpline(clko[0], clko[1])
                resampled_outn = interpn(outp[0])
                resampled_inn  = interpinn(outp[0])
                resampled_inp  = interpinp(outp[0])
                resampled_clko = interpcko(clko[0])
                outn = outp[0], resampled_outn
                inp  = outp[0], resampled_inn
                inn  = outp[0], resampled_inp
                clko = outp[0], resampled_clko

            print("========================inp========================")
            print(inp[1])
            print("========================inn========================")
            print(inn[1])
            print("========================clk========================")
            print(clko[1])

            # we want to cut some off, but leave at least 60-15*2 ??
            CUTOFF = 0#min(max(0, len(outp[0]) - 60), 15)

            step_start_output = outp[1][0] - outn[1][0]
            outdiff = outp[0], outp[1] - outn[1] - step_start_output

            figure, axis = plt.subplots(4,1, sharex=True)
            
            axis[0].plot( outp[0] , outp[1], label = "outp" )
            axis[0].set_title("outp")
            axis[0].set_ylabel('voltage')
            axis[0].grid()
            axis[1].plot( outn[0] , outn[1], label = "outn" )
            axis[1].set_title("outn")
            axis[1].set_ylabel('voltage')
            axis[1].grid()
            axis[2].plot( inn[0], inp[1] - inn[1],   label = "inn" )
            axis[2].set_title("delta")
            axis[2].set_ylabel('voltage')
            axis[2].grid()            
            axis[3].plot( clko[0], clko[1],   label = "clk" )
            axis[3].set_title("clk")
            axis[3].set_ylabel('voltage')
            axis[3].grid()
            figure.suptitle(f'Plot for transient')

            #plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
            #plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')

            plt.show()

            # FLIP
            #outdiff = outdiff[0], -1 * outdiff[1]


            ps, zs = extract_pzs(2, 1, outdiff[0][CUTOFF:], outdiff[1][CUTOFF:])
            list(ps).sort(key=abs)
            zs.sort()
            print(ps)
            print(zs)

            return {'p1' : ps[0], 'p2': ps[1]}


        def post_regression(self, results, data):
            #return {}
            if hasattr(self, 'IS_DEBUG_MODE'):
                for param in results.keys():
                    reg = results[param]

                    y_meas = reg.model.endog
                    y_pred = reg.model.predict(reg.params)

                    plt.scatter(y_meas, y_pred)
                    plt.title(f'Plot for {param}')
                    plt.xlabel('Measured output values')
                    plt.ylabel('Predicted output values based on inputs & model')
                    #plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
                    #plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')
                    plt.grid()
                    plt.show()

            return {}

    tests = [
                SweepTest,
                #DynamicTest
            ]

