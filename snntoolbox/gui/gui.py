# -*- coding: utf-8 -*-
"""
Graphical user interface for SNN toolbox.

Features
--------

    - Allows setting parameters and what tools to use during an experiment.
    - Performs basic checks that specified parameters are valid.
    - Preferences can be saved and reloaded.
    - Tooltips explain the functionality.
    - Automatically recognizes result plots and allows displaying them in a
      separate window.

Created on Mon Apr 11 10:25:53 2016

@author: rbodo
"""

from __future__ import with_statement

import os
import sys
import webbrowser
import json
import threading
from textwrap import dedent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import snntoolbox
from snntoolbox.config import settings, pyNN_settings, update_setup
from snntoolbox.config import model_libs
from snntoolbox.config import simulators, simulators_pyNN
from snntoolbox.gui.tooltip import ToolTip
from snntoolbox.core.pipeline import test_full

if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import filedialog, messagebox, font
    from Queue import Queue
else:
    import tkinter as tk
    from tkinter import filedialog, messagebox, font
    from queue import Queue


class SNNToolboxGUI():
    def __init__(self, root):
        self.initialized = False
        self.root = root
        self.filetypes = (("json files", '*.json'),
                          ("hdf5 files", '*.h5'),
                          ("All files", '*.*'))
        self.default_path_to_pref = os.path.join(snntoolbox._dir,
                                                 'preferences')
        self.define_style()
        self.declare_parameter_vars()
        self.load_settings()
        self.main_container = tk.Frame(root, bg='white')
        self.main_container.pack(side='top', fill='both', expand=True)
        self.globalparams_widgets()
        self.cellparams_widgets()
        self.simparams_widgets()
        self.tools_widgets()
        self.graph_widgets()
        self.top_level_menu()
        self.toggle_state_pyNN(self.settings['simulator'].get())
        self.toggle_percentile_state()
        self.initialized = True

    def define_style(self):
        self.padx = 10
        self.pady = 5
        fontFamily = 'clearlyu devagari'
        self.header_font = (fontFamily, '11', 'bold')
        font.nametofont('TkDefaultFont').configure(family=fontFamily, size=11)
        font.nametofont('TkMenuFont').configure(family=fontFamily, size=11,
                                                weight=font.BOLD)
        font.nametofont('TkTextFont').configure(family=fontFamily, size=11)
        self.kwargs = {'fill': 'both', 'expand': True,
                       'padx': self.padx, 'pady': self.pady}

    def initialize_thread(self):
        self.res_queue = Queue()
        # Create thread for performing the conversion in the background.
        # Make it a daemon so it is killed when the main application is closed.
        self.process_thread = threading.Thread(target=test_full,
                                               args=(self.res_queue,),
                                               name='conversion process',
                                               daemon=True)

    def globalparams_widgets(self):
        # Create a container for individual parameter widgets
        self.globalparams_frame = tk.LabelFrame(self.main_container,
                                                labelanchor='nw',
                                                text="Global parameters",
                                                relief='raised',
                                                borderwidth='3', bg='white')

        self.globalparams_frame.pack(side='left', fill=None, expand=False)
        tip = dedent("""\
              Specify general properties of your model and the steps to
              include in your experiment.""")
        ToolTip(self.globalparams_frame, text=tip, wraplength=750, delay=1499)

        # Dataset
        dataset_frame = tk.Frame(self.globalparams_frame, bg='white')
        dataset_frame.pack(**self.kwargs)
        tk.Button(dataset_frame, text="Load dataset",
                  command=self.set_dataset,
                  font=self.header_font).pack(side='top')
        check_dataset_command = dataset_frame.register(self.check_dataset)
        self.dataset_entry = tk.Entry(
            dataset_frame, textvariable=self.settings['dataset_path'],
            width=20, validate='focusout', bg='white',
            validatecommand=(check_dataset_command, '%P'))
        self.dataset_entry.pack(fill='both', expand=True, side='left')
        scrollX = tk.Scrollbar(dataset_frame, orient=tk.HORIZONTAL,
                               command=self.__scrollHandler)
        scrollX.pack(fill='x', expand=True, side='bottom')
        self.dataset_entry['xscrollcommand'] = scrollX.set
        tip = dedent("""\
            Select a .npy file containing the dataset in form of a tuple
            (X_train, Y_train, X_test, Y_test) of training and test sample
            arrays. With original data of the form
            (channels, num_rows, num_cols), X_train and X_test have dimension
            (num_samples, channels*num_rows*num_cols) for a fully-connected
            network, and (num_samples, channels, num_rows, num_cols) otherwise.
            Y_train and Y_test have dimension (num_samples, num_classes).
            See snntoolbox.io_utils.datasets for examples how to prepare a
            dataset for use in the toolbox.""")
        ToolTip(dataset_frame, text=tip, wraplength=750)

        # Model library
        model_lib_frame = tk.Frame(self.globalparams_frame, bg='white')
        model_lib_frame.pack(**self.kwargs)
        tip = "The neural network library used to create the input model."
        ToolTip(model_lib_frame, text=tip, wraplength=750)
        tk.Label(model_lib_frame, text="Model library", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        model_lib_om = tk.OptionMenu(model_lib_frame,
                                     self.settings['model_lib'],
                                     *list(model_libs))
        model_lib_om.pack(fill='both', expand=True)

        # Percentile
        percentile_frame = tk.Frame(self.globalparams_frame, bg='white')
        percentile_frame.pack(**self.kwargs)
        self.percentile_label = tk.Label(percentile_frame, text="Percentile",
                                         font=self.header_font, bg='white')
        self.percentile_label.pack(fill='both', expand=True)
        self.percentile_sb = tk.Spinbox(
            percentile_frame, bg='white', from_=0, to_=100, increment=1,
            textvariable=self.settings['percentile'], width=10)
        self.percentile_sb.pack(fill='y', expand=True, ipady=5)
        tip = dedent("""\
              Use the activation value in the specified percentile for
              normalization. Set to '50' for the median, '100' for the max.
              Default: '99'.""")
        ToolTip(percentile_frame, text=tip, wraplength=700)

        # Batch size
        batch_size_frame = tk.Frame(self.globalparams_frame, bg='white')
        batch_size_frame.pack(**self.kwargs)
        tk.Label(batch_size_frame, text="Batch size", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        batch_size_sb = tk.Spinbox(batch_size_frame, bg='white',
                                   textvariable=self.settings['batch_size'],
                                   from_=1, to_=1e9, increment=1, width=10)
        batch_size_sb.pack(fill='y', expand=True, ipady=5)
        tip = dedent("""\
              If the builtin simulator 'INI' is used, the batch size specifies
              the number of test samples that will be simulated in parallel.
              Important: When using 'INI' simulator, the batch size can only be
              run using the batch size it has been converted with. To run it
              with a different batch size, convert the ANN from scratch.""")
        ToolTip(batch_size_frame, text=tip, wraplength=700)

        # Test specific samples
        sample_frame = tk.Frame(self.globalparams_frame, bg='white')
        sample_frame.pack(**self.kwargs)
        tk.Label(sample_frame, text="Samples to test:", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        check_sample_command = sample_frame.register(self.check_sample)
        self.sample_entry = tk.Entry(
            sample_frame, bg='white', width=20, validate='key',
            textvariable=self.settings['samples_to_test'],
            validatecommand=(check_sample_command, '%P'))
        self.sample_entry.pack(fill='both', expand=True, side='bottom')
        tip = dedent("""\
              List the indices of specific samples you want to test
              (Don't use brackets or any delimiters other than white spaces).
              """)
        ToolTip(sample_frame, text=tip, wraplength=750)

        # Verbosity
        verbose_frame = tk.Frame(self.globalparams_frame, bg='white')
        verbose_frame.pack(**self.kwargs)
        tk.Label(verbose_frame, text="Verbosity", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        [tk.Radiobutton(verbose_frame, variable=self.settings['verbose'],
                        text=str(i), value=i, bg='white').pack(fill='both',
                                                               side='left',
                                                               expand=True)
         for i in range(4)]
        tip = dedent("""\
              0: No intermediate results or status reports.
              1: Print progress of simulation and intermediate results.
              2: Record spiketrains of all layers for one sample, and save
                 various plots (spiketrains, spikerates, activations,
                 correlations, ...)
              3: Record, plot and return the membrane potential of all layers
                 for the last test sample. Very time consuming. Works only with
                 pyNN simulators.""")
        ToolTip(verbose_frame, text=tip, wraplength=750)

        # Set and display working directory
        path_frame = tk.Frame(self.globalparams_frame, bg='white')
        path_frame.pack(**self.kwargs)
        tk.Button(path_frame, text="Set working dir",
                  command=self.set_cwd, font=self.header_font).pack(side='top')
        check_path_command = path_frame.register(self.check_path)
        self.path_entry = tk.Entry(path_frame,
                                   textvariable=self.settings['path'],
                                   width=20, validate='focusout', bg='white',
                                   validatecommand=(check_path_command, '%P'))
        self.path_entry.pack(fill='both', expand=True, side='left')
        scrollX = tk.Scrollbar(path_frame, orient=tk.HORIZONTAL,
                               command=self.__scrollHandler)
        scrollX.pack(fill='x', expand=True, side='bottom')
        self.path_entry['xscrollcommand'] = scrollX.set
        tip = dedent("""\
              Specify the working directory. There, the toolbox will look for
              ANN models to convert or SNN models to test, load the weights it
              needs and store (normalized) weights.""")
        ToolTip(path_frame, text=tip, wraplength=750)

        # Specify filename base
        filename_frame = tk.Frame(self.globalparams_frame)
        filename_frame.pack(**self.kwargs)
        tk.Label(filename_frame, text="Filename base:", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        check_file_command = filename_frame.register(self.check_file)
        self.filename_entry = tk.Entry(filename_frame, bg='white',
                                       textvariable=self.settings['filename'],
                                       width=20, validate='focusout',
                                       validatecommand=(check_file_command,
                                                        '%P'))
        self.filename_entry.pack(fill='both', expand=True, side='bottom')
        tip = dedent("""\
              Base name of all loaded and saved files during this run. The ANN
              model to be converted is expected to be named '<basename>'.
              The toolbox will save and load converted SNN models under the
              name 'snn_<basename>'. When exporting a converted spiking net to
              test it in a specific simulator, the toolbox writes the exported
              SNN to files named ``snn_<basename>_<simulator>``.""")
        ToolTip(filename_frame, text=tip, wraplength=750)

    def cellparams_widgets(self):
        # Create a container for individual parameter widgets
        self.cellparams_frame = tk.LabelFrame(
            self.main_container, labelanchor='nw', text="Cell\n parameters",
            relief='raised', borderwidth='3', bg='white')
        self.cellparams_frame.pack(side='left', fill=None, expand=False)
        tip = dedent("""\
              Specify parameters of individual neuron cells in the
              converted spiking network. Some are simulator specific.""")
        ToolTip(self.cellparams_frame, text=tip, wraplength=750, delay=1499)

        # Threshold
        v_thresh_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_thresh_frame.pack(**self.kwargs)
        tk.Label(v_thresh_frame, text="v_thresh", bg='white').pack(fill='both',
                                                                   expand=True)
        v_thresh_sb = tk.Spinbox(v_thresh_frame,
                                 textvariable=self.settings['v_thresh'],
                                 from_=-1e3, to_=1e3, increment=1e-3, width=10)
        v_thresh_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Threshold in mV defining the voltage at which a spike is fired."
        ToolTip(v_thresh_frame, text=tip, wraplength=750)

        # Refractory time constant
        tau_refrac_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_refrac_frame.pack(**self.kwargs)
        tk.Label(tau_refrac_frame, text="tau_refrac",
                 bg='white').pack(fill='both', expand=True)
        tau_refrac_sb = tk.Spinbox(tau_refrac_frame,
                                   textvariable=self.settings['tau_refrac'],
                                   width=10, from_=0, to_=1e3, increment=0.01)
        tau_refrac_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Duration of refractory period in milliseconds of the neurons
              after spiking.""")
        ToolTip(tau_refrac_frame, text=tip, wraplength=750)

        # Reset
        v_reset_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_reset_frame.pack(**self.kwargs)
        self.v_reset_label = tk.Label(v_reset_frame, text="v_reset",
                                      state=self.settings['state_pyNN'].get(),
                                      bg='white')
        self.v_reset_label.pack(fill='both', expand=True)
        self.v_reset_sb = tk.Spinbox(
            v_reset_frame, disabledbackground='#eee', width=10,
            textvariable=self.settings['v_reset'], from_=-1e3, to_=1e3,
            increment=0.1, state=self.settings['state_pyNN'].get())
        self.v_reset_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Reset potential in mV of the neurons after spiking."
        ToolTip(v_reset_frame, text=tip, wraplength=750)

        # Resting potential
        v_rest_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_rest_frame.pack(**self.kwargs)
        self.v_rest_label = tk.Label(v_rest_frame, text="v_rest", bg='white',
                                     state=self.settings['state_pyNN'].get())
        self.v_rest_label.pack(fill='both', expand=True)
        self.v_rest_sb = tk.Spinbox(
            v_rest_frame, disabledbackground='#eee', width=10,
            textvariable=self.settings['v_rest'], from_=-1e3, to_=1e3,
            increment=0.1, state=self.settings['state_pyNN'].get())
        self.v_rest_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Resting membrane potential in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(v_rest_frame, text=tip, wraplength=750)

        # e_rev_E
        e_rev_E_frame = tk.Frame(self.cellparams_frame, bg='white')
        e_rev_E_frame.pack(**self.kwargs)
        self.e_rev_E_label = tk.Label(e_rev_E_frame, text="e_rev_E",
                                      state=self.settings['state_pyNN'].get(),
                                      bg='white')
        self.e_rev_E_label.pack(fill='both', expand=True)
        self.e_rev_E_sb = tk.Spinbox(
            e_rev_E_frame, disabledbackground='#eee', width=10,
            textvariable=self.settings['e_rev_E'], from_=-1e-3, to_=1e3,
            increment=0.1, state=self.settings['state_pyNN'].get())
        self.e_rev_E_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Reversal potential for excitatory input in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(e_rev_E_frame, text=tip, wraplength=750)

        # e_rev_I
        e_rev_I_frame = tk.Frame(self.cellparams_frame, bg='white')
        e_rev_I_frame.pack(**self.kwargs)
        self.e_rev_I_label = tk.Label(e_rev_I_frame, text="e_rev_I",
                                      state=self.settings['state_pyNN'].get(),
                                      bg='white')
        self.e_rev_I_label.pack(fill='both', expand=True)
        self.e_rev_I_sb = tk.Spinbox(
            e_rev_I_frame, disabledbackground='#eee', width=10,
            textvariable=self.settings['e_rev_I'], from_=-1e3, to_=1e3,
            increment=0.1, state=self.settings['state_pyNN'].get())
        self.e_rev_I_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Reversal potential for inhibitory input in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(e_rev_I_frame, text=tip, wraplength=750)

        # i_offset
        i_offset_frame = tk.Frame(self.cellparams_frame, bg='white')
        i_offset_frame.pack(**self.kwargs)
        self.i_offset_label = tk.Label(
            i_offset_frame, text="i_offset", bg='white',
            state=self.settings['state_pyNN'].get())
        self.i_offset_label.pack(fill='both', expand=True)
        self.i_offset_sb = tk.Spinbox(i_offset_frame, width=10,
                                      textvariable=self.settings['i_offset'],
                                      from_=-1e3, to_=1e3, increment=1,
                                      state=self.settings['state_pyNN'].get(),
                                      disabledbackground='#eee')
        self.i_offset_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Offset current in nA.
              Only relevant in pyNN-simulators.""")
        ToolTip(i_offset_frame, text=tip, wraplength=750)

        # Membrane capacitance
        cm_frame = tk.Frame(self.cellparams_frame, bg='white')
        cm_frame.pack(**self.kwargs)
        self.cm_label = tk.Label(cm_frame, text="C_mem", bg='white',
                                 state=self.settings['state_pyNN'].get(),)
        self.cm_label.pack(fill='both', expand=True)
        self.cm_sb = tk.Spinbox(cm_frame, textvariable=self.settings['cm'],
                                from_=1e-3, to_=1e3, increment=1e-3, width=10,
                                state=self.settings['state_pyNN'].get(),
                                disabledbackground='#eee')
        self.cm_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Membrane capacitance in nF.
              Only relevant in pyNN-simulators.""")
        ToolTip(cm_frame, text=tip, wraplength=750)

        # tau_m
        tau_m_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_m_frame.pack(**self.kwargs)
        self.tau_m_label = tk.Label(tau_m_frame, text="tau_m", bg='white',
                                    state=self.settings['state_pyNN'].get())
        self.tau_m_label.pack(fill='both', expand=True)
        self.tau_m_sb = tk.Spinbox(tau_m_frame, disabledbackground='#eee',
                                   textvariable=self.settings['tau_m'],
                                   from_=1, to_=1e6, increment=1, width=10,
                                   state=self.settings['state_pyNN'].get())
        self.tau_m_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Membrane time constant in milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_m_frame, text=tip, wraplength=750)

        # tau_syn_E
        tau_syn_E_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_syn_E_frame.pack(**self.kwargs)
        self.tau_syn_E_label = tk.Label(
            tau_syn_E_frame, text="tau_syn_E", bg='white',
            state=self.settings['state_pyNN'].get())
        self.tau_syn_E_label.pack(fill='both', expand=True)
        self.tau_syn_E_sb = tk.Spinbox(tau_syn_E_frame, width=10,
                                       textvariable=self.settings['tau_syn_E'],
                                       from_=1e-3, to_=1e3, increment=1e-3,
                                       state=self.settings['state_pyNN'].get(),
                                       disabledbackground='#eee')
        self.tau_syn_E_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Decay time of the excitatory synaptic conductance in
              milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_syn_E_frame, text=tip, wraplength=750)

        # tau_syn_I
        tau_syn_I_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_syn_I_frame.pack(**self.kwargs)
        self.tau_syn_I_label = tk.Label(
            tau_syn_I_frame, text="tau_syn_I", bg='white',
            state=self.settings['state_pyNN'].get())
        self.tau_syn_I_label.pack(fill='both', expand=True)
        self.tau_syn_I_sb = tk.Spinbox(tau_syn_I_frame, width=10,
                                       textvariable=self.settings['tau_syn_I'],
                                       from_=1e-3, to_=1e3, increment=1e-3,
                                       state=self.settings['state_pyNN'].get(),
                                       disabledbackground='#eee')
        self.tau_syn_I_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Decay time of the inhibitory synaptic conductance in
              milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_syn_I_frame, text=tip, wraplength=750)

    def simparams_widgets(self):
        # Create a container for individual parameter widgets
        self.simparams_frame = tk.LabelFrame(self.main_container,
                                             labelanchor='nw',
                                             text="Simulation\n parameters",
                                             relief='raised',
                                             borderwidth='3', bg='white')
        self.simparams_frame.pack(side='left', fill=None, expand=False)
        tip = dedent("""\
              Specify parameters concerning the simulation of the converted
              spiking network. Some are simulator specific.""")
        ToolTip(self.simparams_frame, text=tip, wraplength=750, delay=1499)

        # Simulator
        simulator_frame = tk.Frame(self.simparams_frame, bg='white')
        simulator_frame.pack(**self.kwargs)
        tip = dedent("""\
            Choose a simulator to run the converted spiking network with.""")
        ToolTip(simulator_frame, text=tip, wraplength=750)
        tk.Label(simulator_frame, text="Simulator", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        simulator_om = tk.OptionMenu(simulator_frame,
                                     self.settings['simulator'],
                                     *list(simulators),
                                     command=self.toggle_state_pyNN)
        simulator_om.pack(fill='both', expand=True)

        # Time resolution
        dt_frame = tk.Frame(self.simparams_frame, bg='white')
        dt_frame.pack(**self.kwargs)
        tk.Label(dt_frame, text="dt", bg='white').pack(fill='x', expand=True)
        dt_sb = tk.Spinbox(dt_frame, textvariable=self.settings['dt'],
                           from_=1e-3, to_=1e3, increment=1e-3, width=10)
        dt_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Time resolution of spikes in milliseconds."
        ToolTip(dt_frame, text=tip, wraplength=750)

        # Duration
        duration_frame = tk.Frame(self.simparams_frame, bg='white')
        duration_frame.pack(**self.kwargs)
        tk.Label(duration_frame, text="duration", bg='white').pack(fill='y',
                                                                   expand=True)
        duration_sb = tk.Spinbox(duration_frame, width=10, increment=1,
                                 from_=self.settings['dt'].get(), to_=1e9,
                                 textvariable=self.settings['duration'])
        duration_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Runtime of simulation of one input in milliseconds."
        ToolTip(duration_frame, text=tip, wraplength=750)

        # Maximum firing rate
        input_rate_frame = tk.Frame(self.simparams_frame, bg='white')
        input_rate_frame.pack(**self.kwargs)
        tk.Label(input_rate_frame, text="input_rate", bg='white').pack(
            fill='both', expand=True)
        input_rate_sb = tk.Spinbox(input_rate_frame,
                                   textvariable=self.settings['input_rate'],
                                   from_=1, to_=10000, increment=1, width=10)
        input_rate_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Poisson spike rate in Hz for a fully-on pixel of input image."
        ToolTip(input_rate_frame, text=tip, wraplength=750)

        # Delay
        delay_frame = tk.Frame(self.simparams_frame, bg='white')
        delay_frame.pack(**self.kwargs)
        self.delay_label = tk.Label(delay_frame, text="delay", bg='white',
                                    state=self.settings['state_pyNN'].get())
        self.delay_label.pack(fill='both', expand=True)
        self.delay_sb = tk.Spinbox(delay_frame, disabledbackground='#eee',
                                   textvariable=self.settings['delay'],
                                   from_=self.settings['dt'].get(), to_=1000,
                                   increment=1, width=10,
                                   state=self.settings['state_pyNN'].get())
        self.delay_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Delay in milliseconds. Must be equal to or greater than the
              resolution.
              Only relevant in pyNN-simulators.""")
        ToolTip(delay_frame, text=tip, wraplength=750)

        # Number of samples to test
        num_to_test_frame = tk.Frame(self.simparams_frame, bg='white')
        num_to_test_frame.pack(**self.kwargs)
        self.num_to_test_label = tk.Label(
            num_to_test_frame, bg='white', text="num_to_test",
            state=self.settings['state_num_to_test'].get())
        self.num_to_test_label.pack(fill='both', expand=True)
        self.num_to_test_sb = tk.Spinbox(
            num_to_test_frame, state=self.settings['state_num_to_test'].get(),
            textvariable=self.settings['num_to_test'], from_=1, to_=1e9,
            increment=1, width=10, disabledbackground='#eee')
        self.num_to_test_sb.pack(fill='y', expand=True, ipady=3)
        tip = dedent("""\
              Number of samples to test.
              Only relevant in pyNN-simulators.""")
        ToolTip(num_to_test_frame, text=tip, wraplength=750)

        # Name of directory where to save plots
        runlabel_frame = tk.Frame(self.simparams_frame, bg='white')
        runlabel_frame.pack(**self.kwargs)
        tk.Label(runlabel_frame, text='run label', bg='white').pack(
            fill='both', expand=True)
        check_runlabel_command = runlabel_frame.register(self.check_runlabel)
        runlabel_entry = tk.Entry(
            runlabel_frame, bg='white', textvariable=self.settings['runlabel'],
            validate='focusout', validatecommand=(check_runlabel_command,
                                                  '%P'))
        runlabel_entry.pack(fill='both', expand=True, side='bottom')
        tip = dedent("""\
            Give your simulation run a name. If verbosity is high, the
            resulting plots will be saved in <cwd>/log/gui/<runlabel>.""")

    def tools_widgets(self):
        self.tools_frame = tk.LabelFrame(self.main_container, labelanchor='nw',
                                         text='Tools', relief='raised',
                                         borderwidth='3', bg='white')
        self.tools_frame.pack(side='left', fill=None, expand=False)
        tip = dedent("""\
              Specify the tools to apply in your experiment.""")
        ToolTip(self.tools_frame, text=tip, wraplength=750, delay=1499)

        # Evaluate ANN
        self.evaluateANN_cb = tk.Checkbutton(
            self.tools_frame, text="Evaluate ANN", bg='white',
            variable=self.settings['evaluateANN'], height=2, width=20)
        self.evaluateANN_cb.pack(**self.kwargs)
        tip = dedent("""\
            If enabled, test the ANN before conversion. If you also enabled
            'normalization' (see parameter 'normalize' below), then the network
            will be evaluated again after normalization.""")
        ToolTip(self.evaluateANN_cb, text=tip, wraplength=750)

        # Normalize
        self.normalize_cb = tk.Checkbutton(
            self.tools_frame, text="Normalize", height=2, width=20,
            bg='white', variable=self.settings['normalize'],
            command=self.toggle_percentile_state)
        self.normalize_cb.pack(**self.kwargs)
        tip = dedent("""\
              Only relevant when converting a network, not during simulation.
              If enabled, the weights of the spiking network will be
              normalized by the highest weight or activation.""")
        ToolTip(self.normalize_cb, text=tip, wraplength=750)

        # Convert ANN
        convert_cb = tk.Checkbutton(self.tools_frame, text="Convert",
                                    variable=self.settings['convert'],
                                    height=2, width=20, bg='white')
        convert_cb.pack(**self.kwargs)
        tip = dedent("""\
              If enabled, load an ANN from working directory (see setting
              'working dir') and convert it to spiking.""")
        ToolTip(convert_cb, text=tip, wraplength=750)

        # Simulate
        simulate_cb = tk.Checkbutton(self.tools_frame, text="Simulate",
                                     variable=self.settings['simulate'],
                                     height=2, width=20, bg='white')
        simulate_cb.pack(**self.kwargs)
        tip = dedent("""\
              If enabled, try to load SNN from working directory (see setting
              'working dir') and test it on the specified simulator (see
              parameter 'simulator').""")
        ToolTip(simulate_cb, text=tip, wraplength=750)

        # Overwrite
        overwrite_cb = tk.Checkbutton(self.tools_frame, text="Overwrite",
                                      variable=self.settings['overwrite'],
                                      height=2, width=20, bg='white')
        overwrite_cb.pack(**self.kwargs)
        tip = dedent("""\
              If disabled, the save methods will ask for permission to
              overwrite files before writing weights, activations, models etc.
              to disk.""")
        ToolTip(overwrite_cb, text=tip, wraplength=750)

        # Start experiment
        self.start_processing_bt = tk.Button(
            self.tools_frame, text="Start", font=self.header_font,
            foreground='#008000', command=self.start_processing,
            state=self.start_state.get())
        self.start_processing_bt.pack(**self.kwargs)
        tip = dedent("""\
              Start processing the steps specified above. Settings can not be
              changed during the run.""")
        ToolTip(self.start_processing_bt, text=tip, wraplength=750)

        # Stop experiment
        self.stop_processing_bt = tk.Button(
            self.tools_frame, text="Stop", font=self.header_font,
            foreground='red', command=self.stop_processing)
        self.stop_processing_bt.pack(**self.kwargs)
        tip = dedent("""\
              Stop the process at the next opportunity. This will usually be
              between steps of normalization, evaluation, conversion and
              simulation.""")
        ToolTip(self.stop_processing_bt, text=tip, wraplength=750)

    def graph_widgets(self):
        # Create a container for buttons that display plots for individual
        # layers.
        if hasattr(self, 'graph_frame'):
            self.graph_frame.pack_forget()
            self.graph_frame.destroy()
        self.graph_frame = tk.Frame(self.main_container, background='white')
        self.graph_frame.pack(side='left', fill=None, expand=False)
        tip = dedent("""\
              Select a layer to display plots like Spiketrains, Spikerates,
              Membrane Potential, Correlations, etc.""")
        ToolTip(self.graph_frame, text=tip, wraplength=750)
        self.select_plots_dir_rb()
        if hasattr(self, 'selected_plots_dir'):
            self.select_layer_rb()

    def select_plots_dir_rb(self):
        self.plot_dir_frame = tk.LabelFrame(self.graph_frame, labelanchor='nw',
                                            text="Select dir", relief='raised',
                                            borderwidth='3', bg='white')
        self.plot_dir_frame.pack(side='top', fill=None, expand=False)
        self.gui_log.set(os.path.join(self.settings['path'].get(),
                                      'log', 'gui'))
        if os.path.isdir(self.gui_log.get()):
            plot_dirs = [d for d in sorted(os.listdir(self.gui_log.get()))
                         if os.path.isdir(os.path.join(self.gui_log.get(), d))]
            self.selected_plots_dir = tk.StringVar(value=plot_dirs[0])
            [tk.Radiobutton(self.plot_dir_frame, bg='white', text=name,
                            value=name, command=self.select_layer_rb,
                            variable=self.selected_plots_dir).pack(
                            fill='both', side='bottom', expand=True)
             for name in plot_dirs]
        open_new_cb = tk.Checkbutton(self.graph_frame, bg='white', height=2,
                                     width=20, text='open in new window',
                                     variable=self.settings['open_new'])
        open_new_cb.pack(**self.kwargs)
        tip = dedent("""\
              If unchecked, the window showing graphs for a certain layer will
              close and be replaced each time you select a layer to plot.
              If checked, an additional window will pop up instead.""")
        ToolTip(open_new_cb, text=tip, wraplength=750)

    def select_layer_rb(self):
        if hasattr(self, 'layer_frame'):
            self.layer_frame.pack_forget()
            self.layer_frame.destroy()
        self.layer_frame = tk.LabelFrame(self.graph_frame, labelanchor='nw',
                                         text="Select layer", relief='raised',
                                         borderwidth='3', bg='white')
        self.layer_frame.pack(side='bottom', fill=None, expand=False)
        self.plots_dir = os.path.join(self.gui_log.get(),
                                      self.selected_plots_dir.get())
        if os.path.isdir(self.plots_dir):
            layer_dirs = [d for d in sorted(os.listdir(self.plots_dir))
                          if d != 'normalization' and
                          os.path.isdir(os.path.join(self.plots_dir, d))]
            [tk.Radiobutton(self.layer_frame, bg='white', text=name,
                            value=name, command=self.display_graphs,
                            variable=self.layer_to_plot).pack(
                            fill='both', side='bottom', expand=True)
             for name in layer_dirs]

    def draw_canvas(self):
        # Create figure with subplots, a canvas to hold them, and add
        # matplotlib navigation toolbar.
        if self.layer_to_plot.get() is '':
            return
        if hasattr(self, 'plot_container') \
                and not self.settings['open_new'].get() \
                and not self.is_plot_container_destroyed:
            self.plot_container.wm_withdraw()
        self.plot_container = tk.Toplevel(bg='white')
        self.plot_container.geometry('800x600')
        self.is_plot_container_destroyed = False
        self.plot_container.wm_title('Results from simulation run {}'.format(
            self.selected_plots_dir.get()))
        self.plot_container.protocol('WM_DELETE_WINDOW', self.close_window)
        tk.Button(self.plot_container, text='Close Window',
                  command=self.close_window).pack()
        f = plt.figure(figsize=(30, 15))
        f.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.99,
                          wspace=0.01, hspace=0.01)
        num_rows = 3
        num_cols = 5
        gs = gridspec.GridSpec(num_rows, num_cols)
        self.a = [plt.subplot(gs[i, 0:-2]) for i in range(3)]
        self.a += [plt.subplot(gs[i, -2]) for i in range(3)]
        self.a += [plt.subplot(gs[i, -1]) for i in range(3)]
        self.canvas = FigureCanvasTkAgg(f, self.plot_container)
        graph_widget = self.canvas.get_tk_widget()
        graph_widget.pack(side='top', fill='both', expand=True)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, graph_widget)

    def close_window(self):
        plt.close()
        self.plot_container.destroy()
        self.is_plot_container_destroyed = True

    def display_graphs(self):
        self.draw_canvas()
        if self.layer_to_plot.get() is '':
            msg = ("Failed to load images. Please select a layer to plot, and "
                   "make sure your working directory contains appropriate "
                   "image files.")
            messagebox.showerror(title="Loading Error", message=msg)
            return
        path_to_plots = os.path.join(self.plots_dir, self.layer_to_plot.get())
        if not os.path.isdir(path_to_plots):
            msg = ("Failed to load images. Please set a working directory "
                   "that contains appropriate image files.")
            messagebox.showerror(title="Loading Error", message=msg)
            return
        saved_plots = sorted(os.listdir(path_to_plots))
        [a.clear() for a in self.a]
        for name in saved_plots:
            i = int(name[:1])
            self.a[i].imshow(mpimg.imread(os.path.join(path_to_plots, name)))

        layer_idx = int(self.layer_to_plot.get()[:2])
        plots_dir_norm = os.path.join(self.plots_dir, 'normalization')
        if os.path.exists(plots_dir_norm):
            normalization_plots = sorted(os.listdir(plots_dir_norm))
        else:
            normalization_plots = []
        activation_distr = None
        weight_distr = None
        for i in range(len(normalization_plots)):
            if int(normalization_plots[i][:2]) == layer_idx:
                activation_distr = normalization_plots[i]
                weight_distr = normalization_plots[i+1]
                break
        if activation_distr and weight_distr:
            self.a[3].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                       'normalization',
                                                       activation_distr)))
            self.a[6].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                       'normalization',
                                                       weight_distr)))
        self.a[-1].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                    'Pearson.png')))
        for a in self.a:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        self.canvas.draw()
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side='left', fill='both', expand=True)

    def top_level_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save preferences",
                             command=self.save_settings)
        filemenu.add_command(label="Load preferences",
                             command=self.load_settings)
        filemenu.add_command(label="Restore default preferences",
                             command=self.restore_default_params)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit_toolbox)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        helpmenu.add_command(label="Documentation", command=self.documentation)
        menubar.add_cascade(label="Help", menu=helpmenu)

    def documentation(self):
        webbrowser.open(os.path.join(sys.exec_prefix, 'docs',
                                     'Documentation.html'))

    def about(self):
        msg = ("This is a collection of tools to convert analog neural "
               "networks to fast and high-performing spiking nets.\n\n"
               "Developed at the Institute of Neuroinformatics, \n"
               "University / ETH Zurich.\n\n"
               "Contact: Bodo Rueckauer \n"
               "bodo.rueckauer@gmail.com \n\n"
               "Version: {} \n\n".format('0.1dev') +
               "2016")
        messagebox.showinfo(title="About SNN Toolbox", message=msg)

    def quit_toolbox(self):
        self.store_last_settings = True
        self.save_settings()
        self.root.destroy()
        self.root.quit()

    def declare_parameter_vars(self):
        # These will be written to disk as preferences.
        self.settings = {'dataset_path': tk.StringVar(),
                         'model_lib': tk.StringVar(),
                         'evaluateANN': tk.BooleanVar(),
                         'normalize': tk.BooleanVar(),
                         'percentile': tk.IntVar(),
                         'convert': tk.BooleanVar(),
                         'simulate': tk.BooleanVar(),
                         'overwrite': tk.BooleanVar(),
                         'batch_size': tk.IntVar(),
                         'verbose': tk.IntVar(),
                         'path': tk.StringVar(value=snntoolbox._dir),
                         'filename': tk.StringVar(),
                         'filename_snn': tk.StringVar(),
                         'filename_snn_exported': tk.StringVar(),
                         'v_thresh': tk.DoubleVar(),
                         'tau_refrac': tk.DoubleVar(),
                         'v_reset': tk.DoubleVar(),
                         'v_rest': tk.DoubleVar(),
                         'e_rev_E': tk.DoubleVar(),
                         'e_rev_I': tk.DoubleVar(),
                         'i_offset': tk.IntVar(),
                         'cm': tk.DoubleVar(),
                         'tau_m': tk.IntVar(),
                         'tau_syn_E': tk.DoubleVar(),
                         'tau_syn_I': tk.DoubleVar(),
                         'dt': tk.DoubleVar(),
                         'simulator': tk.StringVar(),
                         'duration': tk.IntVar(),
                         'input_rate': tk.IntVar(),
                         'delay': tk.IntVar(),
                         'num_to_test': tk.IntVar(),
                         'runlabel': tk.StringVar(),
                         'open_new': tk.BooleanVar(value=True),
                         'log_dir_of_current_run': tk.StringVar(),
                         'state_pyNN': tk.StringVar(value='normal'),
                         'samples_to_test': tk.StringVar(),
                         'state_num_to_test': tk.StringVar(value='normal')}
        # These will not be written to disk as preferences.
        self.is_plot_container_destroyed = True
        self.store_last_settings = False
        self.restore_last_pref = True
        self.layer_rb_set = False
        self.layer_rbs = []
        self.layer_to_plot = tk.StringVar()
        self.start_state = tk.StringVar(value='normal')
        self.stop_state = tk.StringVar(value='normal')
        self.percentile_state = tk.StringVar()
        self.console_output = tk.StringVar()
        self.gui_log = tk.StringVar()

    def restore_default_params(self):
        defaults = settings
        defaults.update(pyNN_settings)
        self.set_preferences(defaults)
        self.toggle_state_pyNN(self.settings['simulator'].get())

    def set_preferences(self, p):
        [self.settings[key].set(p[key]) for key in p]
        if self.settings['path'] == '':
            self.settings['path'] = os.getcwd()

    def save_settings(self):
        s = {key: self.settings[key].get() for key in self.settings}

        if self.store_last_settings:
            if not os.path.exists(self.default_path_to_pref):
                os.makedirs(self.default_path_to_pref)
            with open(os.path.join(self.default_path_to_pref,
                                   '_last_settings.json'), 'w') as f:
                f.write(json.dumps(s))
            self.store_last_settings = False
        else:
            path_to_pref = tk.filedialog.asksaveasfilename(
                defaultextension='.json', filetypes=[("json files", '*.json')],
                initialdir=self.default_path_to_pref,
                title="Choose filename")
            with open(path_to_pref, 'w') as f:
                f.write(json.dumps(s))

    def load_settings(self):
        if self.restore_last_pref:
            self.restore_last_pref = False
            if not os.path.isdir(self.default_path_to_pref):
                return
            path_to_pref = os.path.join(self.default_path_to_pref,
                                        '_last_settings.json')
            if not os.path.isfile(path_to_pref):
                return
        else:
            path_to_pref = tk.filedialog.askopenfilename(
                defaultextension='.json', filetypes=[("json files", '*.json')],
                initialdir=self.default_path_to_pref,
                title="Choose filename")
        s = json.load(open(path_to_pref))
        self.set_preferences(s)

    def start_processing(self):
        if self.settings['filename'].get() == '':
            messagebox.showwarning(title="Warning",
                                   message="Please specify a filename base.")
            return

        self.store_last_settings = True
        self.save_settings()
        self.check_runlabel(self.settings['runlabel'].get())
        update_setup({key: self.settings[key].get() for key in self.settings})

        self.initialize_thread()
        self.process_thread.start()
        self.toggle_start_state(True)
        self.update()

    def stop_processing(self):
        if self.process_thread.is_alive():
            self.res_queue.put('stop')
        self.toggle_stop_state(True)

    def update(self):
        """Update GUI with items from the queue."""
        if self.process_thread.is_alive():
            # Schedule next update
            self.root.after(1000, self.update)
        else:
            # Stop loop of watching process_thread.
            self.toggle_start_state(False)
            self.toggle_stop_state(False)

    def check_sample(self, P):
        if not self.initialized:
            return True
        elif P == '':
            self.toggle_num_to_test_state(True)
            return True
        elif False:
            # Put some other tests here
            return False
        else:
            samples = [int(i) for i in P.split() if i.isnumeric()]
            self.settings['num_to_test'].set(len(samples))
            self.toggle_num_to_test_state(False)
            return True

    def check_file(self, P):
        if not os.path.exists(self.settings['path'].get()) or \
                not any(P in fname for fname in
                        os.listdir(self.settings['path'].get())):
            msg = ("Failed to set filename base:\n"
                   "Either working directory does not exist or contains no "
                   "files with base name \n '{}'".format(P))
            messagebox.showwarning(title="Warning", message=msg)
            return False
        else:
            return True

    def check_path(self, P):
        if not self.initialized:
            return True
        # Look for plots in working directory to display
        self.graph_widgets()

        if not os.path.exists(P):
            msg = "Failed to set working directory:\n" + \
                  "Specified directory does not exist."
            messagebox.showwarning(title="Warning", message=msg)
            return False
        if not any(fname.endswith('.json') for fname in os.listdir(P)):
            msg = "No model file '*.json' found in \n {}".format(P)
            messagebox.showwarning(title="Warning", message=msg)
            return False
        return True

    def check_runlabel(self, P):
        if self.initialized:
            # Set path to plots for the current simulation run
            self.settings['log_dir_of_current_run'].set(
                os.path.join(self.gui_log.get(), P))
            if not os.path.exists(
                    self.settings['log_dir_of_current_run'].get()):
                os.makedirs(self.settings['log_dir_of_current_run'].get())

    def set_cwd(self):
        self.settings['path'].set(filedialog.askdirectory(
            title="Set working directory", initialdir=snntoolbox._dir))
        self.check_path(self.settings['path'].get())
        self.gui_log.set(os.path.join(self.settings['path'].get(),
                                      'log', 'gui'))
        # Look for plots in working directory to display
        self.graph_widgets()

    def set_dataset(self):
        self.settings['dataset_path'].set(filedialog.askopenfilename(
            title="Choose dataset to load", initialdir=snntoolbox._dir,
            defaultextension='npy', multiple=False,
            filetypes=[('all files', '.*'), ('numpy', '.npy')]))

    def check_dataset(self, P):
        if os.path.isfile(P):
            return True
        else:
            messagebox.showerror(title="Error loading dataset",
                                 message="Not a file")
            return False

    def __scrollHandler(self, *L):
        op, howMany = L[0], L[1]
        if op == 'scroll':
            units = L[2]
            self.path_entry.xview_scroll(howMany, units)
        elif op == 'moveto':
            self.path_entry.xview_moveto(howMany)

    def toggle_state_pyNN(self, val):
        if val not in simulators_pyNN:
            self.settings['state_pyNN'].set('disabled')
        else:
            self.settings['state_pyNN'].set('normal')
        for name in pyNN_settings:
            getattr(self, name + '_label').configure(
                state=self.settings['state_pyNN'].get())
            getattr(self, name + '_sb').configure(
                state=self.settings['state_pyNN'].get())

    def toggle_start_state(self, val):
        if val:
            self.start_state.set('disabled')
        else:
            self.start_state.set('normal')
        self.start_processing_bt.configure(state=self.start_state.get())

    def toggle_stop_state(self, val):
        if val:
            self.stop_state.set('disabled')
        else:
            self.stop_state.set('normal')
        self.stop_processing_bt.configure(state=self.stop_state.get())

    def toggle_num_to_test_state(self, val):
        if val and not self.settings['state_pyNN'].get() == 'disabled':
            self.settings['state_num_to_test'].set('normal')
        else:
            self.settings['state_num_to_test'].set('disabled')
        self.num_to_test_label.configure(
            state=self.settings['state_num_to_test'].get())
        self.num_to_test_sb.configure(
            state=self.settings['state_num_to_test'].get())

    def toggle_percentile_state(self):
        if self.settings['normalize'].get():
            self.percentile_state.set('normal')
        else:
            self.percentile_state.set('disabled')
        self.percentile_label.configure(state=self.percentile_state.get())
        self.percentile_sb.configure(state=self.percentile_state.get())


def main():
    root = tk.Tk()
    root.title("SNN Toolbox")
    app = SNNToolboxGUI(root)
    root.protocol('WM_DELETE_WINDOW', app.quit_toolbox)
    root.mainloop()


if __name__ == '__main__':
    main_thread = threading.Thread(target=main, name='main thread')
    main_thread.setDaemon(True)
    main_thread.start()
