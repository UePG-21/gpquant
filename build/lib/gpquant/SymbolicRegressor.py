import random
from typing import Any

import numpy as np
import pandas as pd

from .Backtester import backtester_map
from .Fitness import fitness_map
from .Function import function_map
from .SyntaxTree import SyntaxTree


class SymbolicRegressor:
    def __init__(
        self,
        population_size: int,
        tournament_size: int,
        generations: int,
        stopping_criteria: float,
        p_crossover: float,
        p_subtree_mutate: float,
        p_hoist_mutate: float,
        p_point_mutate: float,
        init_depth: tuple,
        init_method: str,
        function_set: list,
        variable_set: list,
        const_range: tuple,
        ts_const_range: tuple,
        build_preference: list,
        metric: str,
        transformer: str = None,
        transformer_kwargs: dict = None,
        parsimony_coefficient: float = 0,
    ) -> None:
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.p_crossover = p_crossover
        self.p_subtree_mutate = p_subtree_mutate
        self.p_hoist_mutate = p_hoist_mutate
        self.p_point_mutate = p_point_mutate
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = (
            [function_map[f] for f in function_set]
            if function_set
            else list(function_map.values())
        )
        self.variable_set = variable_set
        self.const_range = const_range
        self.ts_const_range = ts_const_range
        self.build_preference = build_preference
        self.metric = fitness_map[metric]
        self.transformer = None if transformer is None else backtester_map[transformer]
        self.transformer_kwargs = transformer_kwargs
        self.parsimony_coefficient = parsimony_coefficient
        self.trees: list[SyntaxTree] = []
        self.fitness: list[float] = []
        self.best_estimator: SyntaxTree = None
        self.best_fitness: float = None

    def __build(self) -> None:
        for i in range(self.population_size):
            self.trees.append(
                SyntaxTree(
                    id=i,
                    init_depth=self.init_depth,
                    init_method=self.init_method,
                    function_set=self.function_set,
                    variable_set=self.variable_set,
                    const_range=self.const_range,
                    ts_const_range=self.ts_const_range,
                    build_preference=self.build_preference,
                    metric=self.metric,
                    transformer=self.transformer,
                    transformer_kwargs=self.transformer_kwargs,
                    parsimony_coefficient=self.parsimony_coefficient,
                )
            )

    def __tournament(self) -> SyntaxTree:
        contenders = random.sample(range(self.population_size), self.tournament_size)
        fitness = [self.fitness[i] for i in contenders]
        if self.metric.sign > 0:
            parent_index = contenders[np.nanargmax(fitness)]
        else:
            parent_index = contenders[np.nanargmin(fitness)]
        return self.trees[parent_index]

    def __evolve(self) -> None:
        offsprings = []
        method_probs = [
            self.p_crossover,
            self.p_subtree_mutate,
            self.p_hoist_mutate,
            self.p_point_mutate,
        ]
        method_probs = np.cumsum(method_probs)
        if method_probs[-1] > 1:
            raise ValueError(
                "sum of crossover and mutation probabilities should <= 1.0"
            )
        for _ in range(self.population_size):
            parent = self.__tournament()
            method = np.searchsorted(method_probs, random.random())
            if method == 0:
                # crossover
                donor = self.__tournament()
                offsprings.append(parent.crossover(donor))
            elif method == 1:
                # subtree mutation
                offsprings.append(parent.subtree_mutate())
            elif method == 2:
                # hoist mutation
                offsprings.append(parent.hoist_mutate())
            elif method == 3:
                # point mutation
                offsprings.append(parent.point_mutate())
            else:
                # reproduction
                offsprings.append(parent.reproduce())
        self.trees = offsprings

    def __log(self, i: int) -> None:
        print(f"------------Generation {str(i + 1).rjust(2)}------------")
        print(f"best estimator: {self.best_estimator}")
        print(f"best fitness: {self.best_fitness}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.__build()
        for i in range(self.generations):
            self.fitness = np.array([tree.fitness(X, y) for tree in self.trees])
            self.best_estimator = self.trees[
                np.nanargmax(self.metric.sign * self.fitness)
            ]
            self.best_fitness = self.metric.sign * np.nanmax(
                self.metric.sign * self.fitness
            )
            self.__log(i)
            if self.metric.sign * (self.best_fitness - self.stopping_criteria) > 0:
                break
            self.__evolve()
    
    def predict(self, X: pd.DataFrame) -> Any:
        return self.best_estimator.execute(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.best_estimator is None:
            raise AttributeError("cannot call score without `self.best_estimator`")
        return self.best_estimator.fitness(X, y)
