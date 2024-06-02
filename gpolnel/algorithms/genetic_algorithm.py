import os
import time
import random
import pickle
import logging
import torch
import pandas as pd
from copy import deepcopy

from gpolnel.algorithms.population_based import PopulationBased
from gpolnel.utils.population import PopulationTree
from gpolnel.utils.inductive_programming import _execute_tree, _get_tree_depth


class GeneticAlgorithm(PopulationBased):
    """Implements Genetic Algorithm (GA).

    Genetic Algorithm (GA) is a meta-heuristic introduced by John
    Holland, strongly inspired by Darwin's Theory of Evolution.
    Conceptually, the algorithm starts with a random-like population of
    candidate-solutions (called chromosomes). Then, by resembling the
    principles of natural selection and the genetically inspired
    variation operators, such as crossover and mutation, the algorithm
    breeds a population of next-generation candidate-solutions (called
    the offspring population), which replaces the previous population
    (a.k.a. the population of parents). The procedure is iterated until
    reaching some stopping criteria, like a predefined number of
    iterations (also called generations).

    An instance of GA can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) a function to initialize the solve at a given point in 𝑆;
        3) a function to select candidate solutions for variation phase;
        4) a function to mutate candidate solutions;
        5) the probability of applying mutation;
        6) a function to crossover two solutions (the parents);
        7) the probability of applying crossover;
        8) the population's size;
        9) the best solution found by the PB-ISA;
        10) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased))
        The best solution found.
    pop_size : int (inherited from PopulationBased)
        The population's size.
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from PopulationBased))
        The initialization procedure.
    selector : function
        The selection procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    p_m : float
        The probability of applying mutation.
    crossover : function
        The crossover procedure.
    p_c : float
        The probability of applying crossover.
    elitism : bool
        A flag which activates elitism during the evolutionary process.
    reproduction : bool
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "GeneticAlgorithm"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.2, p_c=0.8, pop_size=100, elitism=True,
                 reproduction=False, seed=0, device="cpu"):
        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        selector : function
            The selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            The crossover function.
        p_m : float (default=0.2)
            Probability of applying mutation.
        p_c : float (default=0.8)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)  # at this point, it has the pop attribute, but it is None
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.p_c = p_c
        self.elitism = elitism
        self.reproduction = reproduction

    def _set_pop(self, pop_repr):
        """Encapsulates the set method of the population attribute of GeneticAlgorithm algorithm.

        Parameters
        ----------
        pop_repr : list
            A list of solutions' representation.

        Returns
        -------
        None
        """
        # Creates an object of type 'PopulationTree', given the initial representation
        self.pop = PopulationTree(pop_repr)
        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Gets the best in the initial population
        self._set_best_sol()

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0, log_path='./log/', log_xp='GPOL'):
        """Defines the solve procedure of a GA.

        This method implements the following pseudo-code:
            1) Create a random initial population of size n (𝑃);
            2) Repeat until satisfying some termination condition,
             typically the number of generations:
                1) Calculate fitness ∀ individual in 𝑃;
                2) Create an empty population 𝑃’, the population of
                 offsprings;
                3) Repeat until 𝑃’ contains 𝑛 individuals:
                    1) Chose the main genetic operator – crossover,
                     with probability p_c or reproduction with
                     probability (1 − p_c);
                    2) Select two individuals, the parents, by means
                     of a selection algorithm;
                    3) Apply operator selected in 2) 3) 1) to the
                     individuals selected in 2) 3) 2);
                    4) Apply mutation on the resulting offspring with
                     probability p_m;
                    5) Insert individuals from 2) 3) 4) into 𝑃’;
                4) Replace 𝑃 with 𝑃’;
            3) Return the best individual in 𝑃 (the elite).

        Parameters
        ----------
        n_iter : int (default=20)
            The number of iterations.
        tol : float (default=None)
            Minimum required fitness improvement for n_iter_tol
            consecutive iterations to continue the solve. When best
            solution's fitness is not improving by at least tol for
            n_iter_tol consecutive iterations, the solve will be
            automatically interrupted.
        n_iter_tol : int (default=5)
            Maximum number of iterations to continue the solve while
            not meeting the tol improvement.
        start_at : object (default=None)
            The initial starting point in 𝑆 (it is is assumed to be
            feasible under 𝑆's constraints, if any).
        test_elite : bool (default=False)
            Indicates whether assess the best-so-far solution on the
            test partition (this regards SML-based OPs).
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything.
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness.
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness);
                - log = 3: also, writes elite's representation.
        """
        # Optionally, tracks initialization's timing for console's output
        if verbose > 0:
            start = time.time()

        if log > 0:
            logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)

        # 1)
        self._initialize(start_at=start_at)

        # Optionally, evaluates the elite on the test partition
        if test_elite:
            # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
            state = torch.get_rng_state()
            self.pi.evaluate_sol(self.best_sol, train=False, test=True)
            torch.set_rng_state(state)

        # Optionally, computes population's AVG and STD (in terms of fitness)
        if log >= 2 or verbose >= 2:
            self.pop.fit_avg = self.pop.fit.mean().item()
            self.pop.fit_std = self.pop.fit.std().item()

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, self.pop, verbose)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, self.pop, log, log_xp)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        for it in range(1, n_iter + 1, 1):
            # 2) 2)
            offs_pop, start = [], time.time()

            # 2) 3)
            pop_size = self.pop_size - self.pop_size % 2
            while len(offs_pop) < pop_size:
                # 2) 3) 2)
                p1_idx = p2_idx = self.selector(self.pop, self.pi.min_)

                # print('parent indexes {}, {}'.format(p1_idx, p2_idx))

                # Avoids selecting the same parent twice
                while p1_idx == p2_idx:
                    p2_idx = self.selector(self.pop, self.pi.min_)

                # print('parent indexes {}, {}'.format(p1_idx, p2_idx))

                if not self.reproduction:  # performs GP-like variation
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    else:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])
                else:  # performs GA-like variation
                    offs1, offs2 = self.pop[p1_idx], self.pop[p2_idx]
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    if random.uniform(0, 1) < self.p_m:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])

                # 2) 3) 5)
                offs_pop.extend([offs1, offs2])

            # Adds one more individual, if the population size is odd
            if pop_size < self.pop_size:
                offs_pop.append(self.mutator(self.pop[self.selector(self.pop, self.pi.min_)]))

            # If batch training, appends the elite to evaluate_pop it on the same batch(es) as the offspring population
            if self._batch_training:
                offs_pop.append(self.best_sol.repr_)

            # If solutions are objects of type torch.tensor, stacks their representations in the same tensor
            if isinstance(offs_pop[0], torch.Tensor):
                offs_pop = torch.stack(offs_pop)

            # 2) 1)
            offs_pop = globals()[self.pop.__class__.__name__](offs_pop)
            self.pi.evaluate_pop(offs_pop)

            # Overrides elites's information, if it was re-evaluated, and removes it from 'offsprings'
            if self._batch_training:
                self.best_sol.valid = offs_pop.valid[-1]
                self.best_sol.fit = offs_pop.fit[-1]
                # Removes the elite from the object 'offsprings'
                offs_pop.repr_ = offs_pop.repr_[0:-1]
                offs_pop.valid = offs_pop.valid[0: -1]
                offs_pop.fit = offs_pop.fit[0: -1]

            # Performs population's replacement
            if self.elitism:
                offs_pop = self.elite_replacement(offs_pop)
            self.pop = offs_pop
            self._set_best_sol()

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
                state = torch.get_rng_state()
                self.pi.evaluate_sol(self.best_sol, train=False, test=True)
                torch.set_rng_state(state)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log >= 2 or verbose >= 2:
                self.pop.fit_avg = self.pop.fit.mean().item()
                self.pop.fit_std = self.pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, self.pop, log, log_xp)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, self.pop, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)
                if n_iter_bare == n_iter_tol:
                    break
