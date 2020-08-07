import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.position = np.random.uniform(lower_bound, upper_bound)
        self.speed = np.random.uniform(lower_bound-upper_bound, upper_bound-lower_bound)
        self.best_position = self.position
        self.best_value = -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.hyperparams = hyperparams
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.best_global_position = np.zeros(np.size(lower_bound))
        self.best_global_speed = np.zeros(np.size(lower_bound))
        self.best_global_value = -inf
        self.particles = []
        self.counter = 0

        for i in range(self.hyperparams.num_particles):
           self.particles.append(Particle(lower_bound, upper_bound))

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.best_global_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.best_global_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.particles[self.counter].position

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        # Todo: implement

        w = self.hyperparams.inertia_weight
        phip = self.hyperparams.cognitive_parameter
        phig = self.hyperparams.social_parameter

        rp = random.uniform(0.0, 1.0)
        rg = random.uniform(0.0, 1.0)

        particle = self.particles[self.counter]
        self.particles[self.counter].speed = particle.speed * w + phip * rp * (particle.best_position - particle.position) + phig * rg * (self.best_global_position - particle.position)
        self.particles[self.counter].position = particle.position + particle.speed

        self.counter += 1
        if self.counter == self.hyperparams.num_particles:
            self.counter = 0

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        particle = self.particles[self.counter]

        if value > self.best_global_value:
            self.best_global_value = value
            self.best_global_position = particle.position

        if value > particle.best_value:
            self.particles[self.counter].best_value = value
            self.particles[self.counter].best_position = particle.position

        self.advance_generation()