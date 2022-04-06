# Helper Functions
# Qianbo Yin

import numpy as np
import matplotlib.pyplot as plt

def draw_neuron_states(plot_name, voltage, spikes):
    """Draw out the neural states (voltage and spikes) change over time."""
    # time steps
    t_steps = len(voltage)

    # plot the membrane potential
    # TODO: draw the voltage such that there is a discontinous drop at the moment of spike
    fig, (axis_1, axis_2) = plt.subplots(2, sharex=True, num=plot_name)
    axis_1.plot(range(t_steps), voltage)
    axis_1.set_title("Membrane Voltage")
    axis_1.set_ylabel("Voltage (mV)")

    # plot the spikes
    axis_2.plot(range(t_steps), np.zeros(shape=t_steps))
    for spike_pos in np.nonzero(spikes):
        axis_2.vlines(x=spike_pos, ymin=0, ymax=1)
    axis_2.set_title("Spikes")
    axis_2.set_ylabel("Spikes")
    axis_2.set_xlabel("Time Steps")