from ComputationalGraphPrimer import *
import sys,os,os.path
import numpy as np
import re
import operator
import itertools
import math
import random
import torch
from collections import deque
import copy
import matplotlib.pyplot as plt
import networkx as nx

class UpdatedComputationalGraphPrimer(ComputationalGraphPrimer):
    def __init__(self, optimizer_type='nag', **kwargs):
        super().__init__(**kwargs)
        self.optimizer_type = optimizer_type
        # nag init
        if self.optimizer_type == 'nag':
            self.velocity = 0 # weighted average of past gradients initialized to 0

            if 'momentum' in kwargs : momentum = kwargs.pop('momentum')
            self.momentum = momentum if momentum is not None else 0.9 # coefficient determining look ahead distance and how long to keep past gradients
        # rmsprop init
        elif self.optimizer_type == 'rmsprop':
            # larger gradient = smaller learning rate and vice versa to prevent over jumping with gradients
            self.velocity = 0 # weighted average of squared past gradients initialized to 0

            if 'decay' in kwargs : decay = kwargs.pop('decay')
            self.decay = decay if decay is not None else 0.9 # decay rate for past gradients adaptation speed vs. maintaining history

            if 'epsilon' in kwargs : epsilon = kwargs.pop('epsilon')
            self.epsilon = epsilon if epsilon is not None else 1e-9 # small constant to improve stability and avoid division by zero in parameter update step
        # adamw init
        elif self.optimizer_type == 'adamw':
            self.momentum = 0 # running avg of gradients (speed of descent)
            self.velocity = 0 # running average of squared gradients (scaling lr to not over/under commit)

            if 'decay1' in kwargs : decay1 = kwargs.pop('decay1')
            self.decay1 = decay1 if decay1 is not None else 0.9

            if 'decay2' in kwargs : decay2 = kwargs.pop('decay2')
            self.decay2 = decay2 if decay2 is not None else 0.999

            # bias bc momentum and velocity both initialized to 0
            self.momentum_bias = self.momentum / (1-self.decay1)
            self.velocity_bias = self.velocity / (1-self.decay2)

            if 'lamb' in kwargs : lamb = kwargs.pop('lamb')
            self.lamb = lamb if lamb is not None else 0.9 # regularization parameter

            if 'epsilon' in kwargs : epsilon = kwargs.pop('epsilon')
            self.epsilon = epsilon if epsilon is not None else 1e-9 # small constant to improve stability and avoid division by zero in parameter update step
