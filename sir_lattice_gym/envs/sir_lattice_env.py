import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import bokeh.plotting
import bokeh.io
bokeh.io.output_notebook()

class SIRLatticeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, GRID_SIZE=10, I_0=3, c=0.2, neighbor_r=1):
        self.grid_size = GRID_SIZE
        self.I_0 = I_0 #Number of initial infected
        self.c = c  #Parameter depicting infectiousness period
        self.grid = None
        self.neighbor_r = neighbor_r #Neighbor radius
        self.time = 0
        self.trajectory = []
        self.test_number = 30
        self.done = 0

        self.action_space = spaces.MultiDiscrete([2 for i in range(self.grid_size**2)])
        self.observation_space = spaces.MultiDiscrete([3 for i in range(self.grid_size**2)])

        self.reset()
        
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        #Penalizing the number of tests over a threshold
        cost = ((sum(action) - self.test_number)*(sum(action) - self.test_number>0))**2
        
        #Attaining the number of susceptibles (0's)
        self.S = len(np.where(self.grid == 0)[0])

        #I_position_tuples is a tuple of 2 lists; one containing x loc, y loc in other
        I_position_tuple = np.where(self.grid == 1)
        
        #Finding the number of infectious 
        self.I = len(I_position_tuple[0])
        #Keeping a tally for the number of susceptibles and infectious
        self.trajectory.append([self.S, self.I])
        
        obs = deepcopy(action)

        #Stacking the action list for later use
        action = [action[i:i+self.grid_size] for i in range(0, self.grid_size**2, self.grid_size)]
        
        #For every infectious, I am either going to infect others or recover the infectious 
        for i in range(self.I):
            x, y = I_position_tuple[0][i], I_position_tuple[1][i]
            #Criteria is if I draw a number less than a threshold or if I test the individual while
            #they are infectious then I make them a recovered, else I infect a neighbor
            if random.random() < self.c or action[x][y] == 1:
                self.grid[x][y] = 2
                obs[x * self.grid_size + y] = 2
            else:
                self.infect_neighbor(x, y)
        
        #I_position_tuples is a tuple of 2 lists; one containing x loc, y loc in other
        I_position_tuple = np.where(self.grid == 1)
        self.I = len(I_position_tuple[0])
        
        self.time += 1
        if self.time > 200:
            self.done = 1
        #If all are recovered, end simulation
        if np.all(self.grid[self.grid>0] == self.grid[self.grid == 2]):
            self.done = 1
        
        return obs, -(self.I + cost), self.done, {}
    
    def infect_neighbor(self, x, y):
        S_neighbors = []
        #Find all the susceptible neighbors and add them to S_neighbors
        for i in range(x - self.neighbor_r, x + self.neighbor_r + 1):
            for j in range(y - self.neighbor_r, y + self.neighbor_r + 1):
                if i < self.grid_size and i >= 0 and j < self.grid_size and j >= 0:
                    if self.grid[i, j] == 0:
                        S_neighbors.append((i,j))
        
        #Infect one of the neighbors grant at least one neighbor is susceptible
        if len(S_neighbors) > 0:
            next_infected = random.choice(S_neighbors)
            self.grid[next_infected[0], next_infected[1]] = 1
    
    def show_trajectory(self):
        plot = bokeh.plotting.figure(plot_width=600,
                               plot_height=400,
                               x_axis_label='time')
        colors = ['blue', 'red', 'orange']
        labels = ['Susceptible', 'Infected', 'Recovered']
        for i in range(2):
            plot.line(range(len(self.trajectory)), [item[i] for item in self.trajectory], color=colors[i], legend=labels[i])
        plot.line(range(len(self.trajectory)), [self.grid_size**2 - sum(item) for item in self.trajectory], color=colors[2], legend=labels[2])
        return bokeh.plotting.show(plot)
    
    def reset(self):
        #Reset grid to have infected(1's) at random locations in grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.I_0):
            self.grid[random.randrange(self.grid_size)][random.randrange(self.grid_size)] = 1 
        self.time = 0
        self.done = 0
        self.trajectory = []
        return np.zeros((self.grid_size**2))
    
    def render(self):
        plot = bokeh.plotting.figure(x_range=(0, self.grid_size), y_range=(0, self.grid_size))
        plot.image(image=[self.grid], x=0, y=0, dw=self.grid_size, dh=self.grid_size, palette=bokeh.palettes.Viridis256)
        return bokeh.plotting.show(plot)
    
    def show(self):
        plot = bokeh.plotting.figure(x_range=(0, self.grid_size), y_range=(0, self.grid_size))
        plot.image(image=[self.grid], x=0, y=0, dw=self.grid_size, dh=self.grid_size, palette=bokeh.palettes.Viridis256)
        return bokeh.plotting.show(plot)
    
    def close(self):
        pass
