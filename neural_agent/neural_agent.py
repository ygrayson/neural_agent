# Simplest neural agent possible, for fun
# Qianbo Yin

import logging
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

class Neuron():
    def __init__(self, neuron_type = 'LIF', config={'dt': 0.125}):
        # Simulation config
        self.t_refrac = 0  # remaining refractory time
        self.refrac_period = 0.125  # refractory period (ms)
        self.config = config

        # Neuron Properties
        self.type = neuron_type
        self.V_rest = 0.0  # Neuron resting potential (mV)
        self.Vm = 0.0  # Neuron voltage potential (mV)
        self.Rm = 1  # Resistance (kOhm)
        self.Cm = 10  # Capacitance (uF)
        self.tau_m = self.Rm * self.Cm  # Time constant
        self.Vth = 0.75  # spike threshold (mV), = 1 or 0.75

        logging.debug("Created {} neuron with threshold {} mV, refractory time {:.3f} and step time {} ms".format(
                      self.type, self.Vth, self.refrac_period, self.config['dt']))
    
    def neural_step(self, neuron_input):
        """one time step simulated for the neuron given the input current."""
        # logging
        logging.debug("Neuron.neural_step called with input {:.1f} mA, current voltage {:.5f} mV, refractory time {} ms remaining"
                .format(neuron_input, self.Vm, self.t_refrac))
        
        # spike indicator
        spike = 0

        if self.t_refrac == 0:
            # update voltage value for current time
            self.Vm += (-(self.Vm - self.V_rest) + self.Rm * neuron_input) / self.tau_m * self.config['dt']
            if self.Vm >= self.Vth:
                # a spike is generated and refractory period starts
                spike = 1
                self.Vm = self.V_rest
                self.t_refrac = self.refrac_period
                logging.debug("Spike!")
        else:
            # do nothing if neuron in refractory period
            self.t_refrac -= self.config['dt']
            if self.t_refrac < 0:
                self.t_refrac = 0

        return spike

    
    def spike_generator(self, neuron_input):
        """Given a input current over time, generate spike train."""
        pass
        
    
class Simple_agent():
    """simple agent."""
    def __init__(self, init_state=0):
        """combine all neurons into an agent acting as a whole."""
        logging.info("Agent created...")

        # agent configuration parameters
        self.state = init_state
        self.time = 0
        self.dt = 0.125 # simulation time step (ms)

        # agent neurons
        self.left_sense_neuron = Neuron(neuron_type='LIF', config={'dt': self.dt})
        self.right_sense_neuron = Neuron(neuron_type='LIF', config={'dt': self.dt})
        self.left_motor_neuron = Neuron(neuron_type='LIF', config={'dt': self.dt})
        self.right_motor_neuron = Neuron(neuron_type='LIF', config={'dt': self.dt})

    
    def step(self, env_force, boundary_val):
        """step function."""
        # state changed by external force
        self.state += env_force
        self.time += self.dt
        ls_spike, rs_spike, lm_spike, rm_spike = 0, 0, 0, 0 # 4 neurons' spike or not
        logging.debug("Current time: {:.3f} ms, Current state: {:.2f}".format(self.time, self.state))

        # hit the right wall and activate right sensory neuron
        if self.state >= boundary_val:
            # right sensory neuron activated
            current = 10 * (self.state - boundary_val + 1)
            rs_spike = self.right_sense_neuron.neural_step(current)

            # left motor neuron receive right sensory neuron's spikes (if any)
            lm_spike = self.left_motor_neuron.neural_step(10 * rs_spike)

            # when left motor neuron activated, state changes
            if lm_spike:
                self.state -= 1
        # hit the left wall and activate left sensory neuron
        elif self.state <= -boundary_val:
            # left sensory neuron activated
            current = 10 * (boundary_val - self.state + 1)
            ls_spike = self.left_sense_neuron.neural_step(current)

            # right motor neuron receive left sensory neuron's spikes (if any)
            rm_spike = self.right_motor_neuron.neural_step(10 * ls_spike)

            # when right motor neuron activated, state changes
            if rm_spike:
                self.state += 1
        else:
            # nothing happens if the agent is in the middle
            pass
        return (self.state, self.left_sense_neuron.Vm, self.right_sense_neuron.Vm, self.left_motor_neuron.Vm, self.right_motor_neuron.Vm, 
                ls_spike, rs_spike, lm_spike, rm_spike)
        

class Environment():
    """The overall environment of the world."""
    def __init__(self, t_steps=1000, wind=0.5, boundary=10) -> None:
        """
        currently only 1 simple agent in this environment.
        wind: environmental force blowing wind
        boundary: two sides of boundaries that will activate agent's sensory neuron
        """
        logging.info("Environment created...")

        # agent and environment parameters
        self.agent = Simple_agent(init_state=0)
        self.t_steps = t_steps
        self.wind = wind
        self.boundary = boundary

        # records
        self.records = {
            'agent_state': np.empty(shape=self.t_steps), 
            'left_sense_voltage': np.empty(shape=self.t_steps), 
            'right_sense_voltage': np.empty(shape=self.t_steps), 
            'left_motor_voltage': np.empty(shape=self.t_steps), 
            'right_motor_voltage': np.empty(shape=self.t_steps), 
            'left_sense_spike': np.empty(shape=self.t_steps), 
            'right_sense_spike': np.empty(shape=self.t_steps), 
            'left_motor_spike': np.empty(shape=self.t_steps), 
            'right_motor_spike': np.empty(shape=self.t_steps)
            }
    
    def static_draw_out(self):
        """draw out the agent trajectory, neuron states etc."""

        # Agent trajectory plot
        plt.figure(num='Agent Trajectory')
        plt.plot(self.records['agent_state'], range(self.t_steps))
        plt.title("Agent Trajectory")
        plt.vlines(-2, ymin=0, ymax=self.t_steps, colors='r', linestyles='dashed')
        plt.vlines(2, ymin=0, ymax=self.t_steps, colors='r', linestyles='dashed')
        plt.xlim([min(-2, min(self.records['agent_state'])) - 1, max(2, max(self.records['agent_state'])) + 1])
        plt.xlabel("Position")
        plt.ylabel("Time (0.125ms)")
        plt.text(1, self.t_steps + 1, 'boundary')
        plt.text(-3, self.t_steps + 1, 'boundary')
        plt.text(5, self.t_steps + 1, 'wind -----> 0.2 unit/t_step')

        # Agent neuron plots
        draw_neuron_states('Left Sensory Neuron', self.records['left_sense_voltage'], self.records['left_sense_spike'])
        draw_neuron_states('Left Motor Neuron', self.records['left_motor_voltage'], self.records['left_motor_spike'])
        draw_neuron_states('Right Sensory Neuron', self.records['right_sense_voltage'], self.records['right_sense_spike'])
        draw_neuron_states('Right Motor Neuron', self.records['right_motor_voltage'], self.records['right_motor_spike'])
        # logging.debug(self.records['left_motor_voltage'])
        # logging.debug(self.records['right_sense_voltage'])
        # logging.debug(self.records['right_sense_spike'])
        plt.show()

    def dynamic_draw_out(self):
        """draw out the agent trajectory, neuron states in real time."""
        # create mosaic plots
        whole_plot = [['trajectory', 'lsv', 'rsv'],
                      ['trajectory', 'lss', 'rss'],
                      ['.', 'lmv', 'rmv'],
                      ['.', 'lms', 'rms']] # lsv - left sensory voltage, similar for the rest
        fig, axes = plt.subplot_mosaic(whole_plot, constrained_layout=True, figsize=(6.4*3, 4.8*2), num='Agent Simulation')
        
        # trajectory plot preliminary
        axes['trajectory'].set_title("Agent Trajectory")
        axes['trajectory'].vlines(-2, ymin=0, ymax=self.t_steps, colors='r', linestyles='dashed')
        axes['trajectory'].vlines(2, ymin=0, ymax=self.t_steps, colors='r', linestyles='dashed')
        axes['trajectory'].set_xlim([min(-3, min(self.records['agent_state'])) - 1, max(3, max(self.records['agent_state'])) + 1])
        axes['trajectory'].set_xlabel("Position")
        axes['trajectory'].set_ylabel("Time (0.125ms)")
        axes['trajectory'].text(1, self.t_steps + 1, 'boundary')
        axes['trajectory'].text(-3, self.t_steps + 1, 'boundary')
        axes['trajectory'].text(-3, self.t_steps + 15, 'wind -----> 0.2 unit/t_step')

        # left sensory and motor neuron state plot preliminary
        axes['lsv'].set_title("Left Sensory Neuron")
        axes['lsv'].set_ylabel("Membrane Voltage (mV)")
        axes['lsv'].set_xlim([0, self.t_steps])
        axes['lsv'].set_ylim([-0.1, 1.0])
        axes['lsv'].set_xticklabels([])
        axes['lsv'].hlines(y=self.agent.left_sense_neuron.Vth, xmin=0, xmax=self.t_steps, colors='r', linestyles='dashed', linewidths=1.0)
        axes['lss'].set_ylabel("Spikes")
        axes['lss'].set_xlabel("Time (0.125ms)")
        axes['lss'].set_xlim([0, self.t_steps])
        axes['lss'].set_ylim([-0.1, 1.2])

        axes['lmv'].set_title("Left Motor Neuron")
        axes['lmv'].set_ylabel("Membrane Voltage (mV)")
        axes['lmv'].set_xlim([0, self.t_steps])
        axes['lmv'].set_ylim([-0.1, 1.0])
        axes['lmv'].set_xticklabels([])
        axes['lmv'].hlines(y=self.agent.left_motor_neuron.Vth, xmin=0, xmax=self.t_steps, colors='r', linestyles='dashed', linewidths=1.0)
        axes['lms'].set_ylabel("Spikes")
        axes['lms'].set_xlabel("Time (0.125ms)")
        axes['lms'].set_xlim([0, self.t_steps])
        axes['lms'].set_ylim([-0.1, 1.2])

        # right sensory and motor neuron state plot preliminary
        axes['rsv'].set_title("Right Sensory Neuron")
        axes['rsv'].set_ylabel("Membrane Voltage (mV)")
        axes['rsv'].set_xlim([0, self.t_steps])
        axes['rsv'].set_ylim([-0.1, 1.0])
        axes['rsv'].set_xticklabels([])
        axes['rsv'].hlines(y=self.agent.right_sense_neuron.Vth, xmin=0, xmax=self.t_steps, colors='r', linestyles='dashed', linewidths=1.0)
        axes['rss'].set_ylabel("Spikes")
        axes['rss'].set_xlabel("Time (0.125ms)")
        axes['rss'].set_xlim([0, self.t_steps])
        axes['rss'].set_ylim([-0.1, 1.2])

        axes['rmv'].set_title("Right Motor Neuron")
        axes['rmv'].set_ylabel("Membrane Voltage (mV)")
        axes['rmv'].set_xlim([0, self.t_steps])
        axes['rmv'].set_ylim([-0.1, 1.0])
        axes['rmv'].set_xticklabels([])
        axes['rmv'].hlines(y=self.agent.right_motor_neuron.Vth, xmin=0, xmax=self.t_steps, colors='r', linestyles='dashed', linewidths=1.0)
        axes['rms'].set_ylabel("Spikes")
        axes['rms'].set_xlabel("Time (0.125ms)")
        axes['rms'].set_xlim([0, self.t_steps])
        axes['rms'].set_ylim([-0.1, 1.2])

        # start plotting in real time
        for t in range(self.t_steps):
            axes['trajectory'].plot(self.records['agent_state'][0: t], range(t))

            # left sensory neuron
            axes['lsv'].plot(range(t), self.records['left_sense_voltage'][0: t])
            axes['lss'].plot(range(t+1), np.zeros(shape=t+1), 'b-')
            if self.records['left_sense_spike'][t] == 1:
                axes['lss'].vlines(x=t, ymin=0, ymax=1, colors='b')
            
            # left motor neuron
            axes['lmv'].plot(range(t), self.records['left_motor_voltage'][0: t])
            axes['lms'].plot(range(t+1), np.zeros(shape=t+1), 'b-')
            if self.records['left_motor_spike'][t] == 1:
                axes['lms'].vlines(x=t, ymin=0, ymax=1, colors='b')

            # right sensory neuron
            axes['rsv'].plot(range(t), self.records['right_sense_voltage'][0: t])
            axes['rss'].plot(range(t+1), np.zeros(shape=t+1), 'b-')
            if self.records['right_sense_spike'][t] == 1:
                axes['rss'].vlines(x=t, ymin=0, ymax=1, colors='b')

            # right motor neuron
            axes['rmv'].plot(range(t), self.records['right_motor_voltage'][0: t])
            axes['rms'].plot(range(t+1), np.zeros(shape=t+1), 'b-')
            if self.records['right_motor_spike'][t] == 1:
                axes['rms'].vlines(x=t, ymin=0, ymax=1, colors='b')

            plt.pause(0.01)
        plt.show()
    
    def pygame_sim(self):
        """Pygame simulation of the agent in real time."""
        # set up pygame and window
        pygame.init()
        screen = pygame.display.set_mode([800, 800])

        # Run until the user asks to quit
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # draw the agent and environment in real time
            for t in range(self.t_steps):
                # Fill the background with white
                screen.fill((255, 255, 255))

                # Draw the boundary of environment
                pygame.draw.line(screen, color=(255, 0, 0), start_pos=(300, 0), end_pos=(300, 800), width=2)
                pygame.draw.line(screen, color=(255, 0, 0), start_pos=(500, 0), end_pos=(500, 800), width=2)

                # draw the agent
                pos = self.records['agent_state'][t]
                pygame.draw.circle(screen, color=(0, 0, 255), center=(pos*40 + 400, 400), radius=15)

                # display
                pygame.display.flip()
                time.sleep(0.05)
            running = False

        # Done! Time to quit.
        pygame.quit()
    
    def simulate(self):
        """start simulating the environment with whatever agent in it."""
        logging.info("Starting environment simulation...")
        
        # loop through and start to simulate
        for t in range(self.t_steps):
            record = self.agent.step(self.wind, self.boundary)
            agent_state, ls_voltage, rs_voltage, lm_voltage, rm_voltage, ls_spike, rs_spike, lm_spike, rm_spike = record
            self.records['agent_state'][t] = agent_state
            self.records['left_sense_voltage'][t] = ls_voltage
            self.records['right_sense_voltage'][t] = rs_voltage
            self.records['left_motor_voltage'][t] = lm_voltage
            self.records['right_motor_voltage'][t] = rm_voltage
            self.records['left_sense_spike'][t] = ls_spike
            self.records['right_sense_spike'][t] = rs_spike
            self.records['left_motor_spike'][t] = lm_spike
            self.records['right_motor_spike'][t] = rm_spike
        
        # show
        logging.debug("Agent trajectory is: {}".format(self.records['agent_state']))
        # self.dynamic_draw_out()
        self.pygame_sim()
        

def main():
    """main function to start the environment to run."""
    # logging level
    logging.basicConfig(level=logging.INFO)
    logging.info("----------Simulation begins----------")

    # agent-environment simulation
    total_t_steps = 1000 # one unit of t_step is the value of dt, which defaults to 0.125ms
    env = Environment(t_steps=total_t_steps, wind=0.1, boundary=2)
    env.simulate()

    logging.info("----------Simulation completed----------")


if __name__ == '__main__':
    main()