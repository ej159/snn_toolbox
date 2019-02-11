# -*- coding: utf-8 -*-
"""Building and simulating spiking neural networks using SpiNNaker.
Everything is done by PyNN but loading and saving is disabled in utils as per Brian2
"""
from __future__ import division
import warnings
from future import standard_library
import numpy as np
# noinspection PyUnresolvedReferences
from snntoolbox.simulation.target_simulators.pyNN_target_sim import SNN as PyNNSNN

standard_library.install_aliases()

class SNN(PyNNSNN):

    '''def save(self, path, filename):
        warnings.warn("Saving SpiNNaker spiking model to disk is not yet implemented.", RuntimeWarning)

    def load(self, path, filename):
        raise NotImplementedError("Loading SpiNNaker spiking model from disk is not yet implemented.")
    '''
    def init_cells(self):

        vars_to_record = self.get_vars_to_record()

        if 'spikes' in vars_to_record:
            self.layers[0].record([str('spikes')])  # Input layer has no 'v'

        for layer in self.layers[1:]:
            # Commenting this section out means everything runs but no spikes
            layer.initialize(v=self.layers[1].get('v_rest')[0])
            layer.record(vars_to_record)

        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess).
        if 'spikes' not in vars_to_record:
            vars_to_record.append(str('spikes'))
            self.layers[-1].record(vars_to_record)        
        
    def simulate(self, **kwargs):
        if self._poisson_input:
            rates = kwargs[str('x_b_l')].flatten()
            rates = rates / self.rescale_fac * 1000
            self.layers[0].set(rate=rates)
        elif self._dataset_format == 'aedat':
            raise NotImplementedError
        else:
            constant_input_currents = kwargs[str('x_b_l')].flatten()
            try:
                for neuron_idx, neuron in enumerate(self.layers[0]):
                    # TODO: Implement constant input currents.
                    neuron.current = constant_input_currents[neuron_idx]
            except AttributeError:
                raise NotImplementedError

        self.sim.set_number_of_neurons_per_core(self.sim.IF_curr_exp, 32)
        self.sim.run(self._duration - self._dt)
        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t
    
    def get_vars_to_record(self):
        """Get variables to record during simulation.

        Returns
        -------

        vars_to_record: list[str]
            The names of variables to record during simulation.
        """

        vars_to_record = []

        if any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                'hist_spikerates_activations'} & self._plot_keys) \
                or 'spiketrains_n_b_l_t' in self._log_keys:
            vars_to_record.append(str('spikes'))

        if 'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys:
            vars_to_record.append(str('v'))
            

        return vars_to_record   
    
            
    def get_spiketrains_input(self):
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        spiketrains_flat = self.layers[0].get_data().segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t

    def get_spiketrains_output(self):
        shape = [self.batch_size, self.num_classes, self._num_timesteps]
        spiketrains_flat = self.layers[-1].get_data().segments[-1].spiketrains
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(spiketrains_flat,
                                                               shape)
        return spiketrains_b_l_t