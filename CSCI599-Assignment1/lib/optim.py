from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for layer in self.net.layers:
            for n, dv in layer.grads.items():

                if n not in self.velocity: # First iteration
                    self.velocity[n] = 0.0

                v = self.momentum*self.velocity[n] - self.lr*dv  # Calc
                
                layer.params[n] += v   # Update
                
                self.velocity[n] = v   # Save

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for layer in self.net.layers:
            for n, dv in layer.grads.items():

                if n not in self.cache: # First iteration
                    self.cache[n] = 0.0

                m = self.decay*self.cache[n] + (1-self.decay)*(dv**2)

                layer.params[n] -= self.lr*dv/((m + self.eps)**0.5) # Update

                self.cache[n] = m # Save
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        for layer in self.net.layers:
            for n, dv in layer.grads.items():

                self.t += 1

                if n not in self.mt: # First iteration
                    self.mt[n] = 0.0
                    self.vt[n] = 0.0

                m = self.beta1*self.mt[n] + (1-self.beta1)*dv 
                v = self.beta2*self.vt[n] + (1-self.beta2)*(dv**2)

                mcorr = m/(1-(self.beta1**self.t))
                vcorr = v/(1-(self.beta2**self.t))

                layer.params[n] -= (self.lr*mcorr)/(vcorr**0.5 + self.eps)

                self.mt[n] = m
                self.vt[n] = v

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
