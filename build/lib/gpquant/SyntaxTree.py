import random
from typing import Any
from copy import deepcopy

import numpy as np
import pandas as pd

from .Backtester import Backtester
from .Fitness import Fitness
from .Function import Function


class Node:
    def __init__(self, data, is_ts: float = False):
        self.data = data  # Function (function), str (variable) or int (constant)
        self.parent: "Node" = None
        self.children: list[Node] = []  # nodes in order of function arguments
        self.is_ts = is_ts  # if data of this node is time-series const

    def __str__(self):
        if self.children:
            if isinstance(self.data, Function):
                child_str = [str(node) for node in self.children]
                return f'{self.data.name}({", ".join(child_str)})'
            else:
                return f"{self.data}"
        else:
            if isinstance(self.data, Function):
                return self.data.name
            else:
                return f"{self.data}"

    def __call__(self, X: pd.DataFrame):
        if isinstance(self.data, Function):
            fixed_var = [X[param] for param in self.data.fixed_params]
            free_var = [node(X) for node in self.children]
            return self.data(*fixed_var, *free_var)
        elif isinstance(self.data, str):
            return X[self.data]
        elif self.is_ts:
            return self.data
        else:
            # input except time-series constant must be a vector
            return np.full(len(X), self.data)

    def add_child(self, child: "Node"):
        # always insert to the front
        self.children.insert(0, child)
        # also set parent of child
        child.parent = self

    def chg_child(self, old_child: "Node", new_child: "Node"):
        index = self.children.index(old_child)
        self.children.remove(old_child)
        # also delete parent of old child
        self.children.insert(index, new_child)
        # also set parent of new child
        new_child.parent = self


class SyntaxTree:
    def __init__(
        self,
        id: int,
        init_depth: tuple,
        init_method: str,
        function_set: list,
        variable_set: list,
        const_range: tuple,
        ts_const_range: tuple,
        build_preference: list,
        metric: Fitness = None,
        transformer: Backtester = None,
        transformer_kwargs: dict = None,
        parsimony_coefficient: float = 0,
    ) -> None:
        """
        @param id: tree id, the data in the parent node of root for locating it
        @param init_depth: range of max depth
        @param init_method: building method: 'full', 'grow' or 'half and half'
        @param function_set: available functions
        @param variable_set: available variables
        @param const_range: available constant range
        @param ts_const_range: available time-series constant range
        @param build_preference: probability to choose function over terminal and variable over constant
        @param metric: fitness metric
        @param transformer: backtester to transform factor into asset
        @param transformer_kwargs: arguments of transformer
        @param parsimony_coefficient: penalty strength for formula inflation
        """
        self.id = Node(id)
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.variable_set = variable_set
        self.const_range = const_range
        self.ts_const_range = ts_const_range
        self.build_preference = build_preference
        self.metric = metric
        self.transformer = transformer
        self.transformer_kwargs = transformer_kwargs
        self.parsimony_coefficient = parsimony_coefficient
        self.ttl_shift = 0  # int; sum of d as time-series constant
        self.__build()
        self.nodes = self.__flatten()  # list; flattened tree

    def __str__(self) -> str:
        return str(self.nodes[0])

    def __len__(self) -> int:
        return len(self.nodes)

    def __build(self) -> None:
        # set max depth and build method
        max_depth = random.randint(*self.init_depth)
        if self.init_method == "half and half":
            method = "full" if random.random() < 0.5 else "grow"
        else:
            method = self.init_method

        # set root node data to a function
        data = random.choice(self.function_set)
        node = Node(data)
        self.id.add_child(node)
        parent_stack = [node]
        children_stack = [data.arity - len(data.fixed_params)]

        # continue adding child
        while children_stack:
            depth = len(children_stack)
            # first, check if need to add time series constant
            ts_const_num = parent_stack[-1].data.is_ts
            while ts_const_num:
                data = random.randint(*self.ts_const_range)
                node = Node(data, is_ts=True)
                parent_stack[-1].add_child(node)
                ts_const_num -= 1
                children_stack[-1] -= 1
                while children_stack[-1] == 0:
                    children_stack.pop()
                    if not children_stack:
                        return None
                    children_stack[-1] -= 1
                    parent_stack.pop()
            # second, determine to add function or terminal
            if depth < max_depth and (
                method == "full" or random.random() < self.build_preference[0]
            ):
                # add function
                data = random.choice(self.function_set)
                node = Node(data)
                parent_stack[-1].add_child(node)
                parent_stack.append(node)
                children_stack.append(data.arity - len(data.fixed_params))
            else:
                # add terminal, and determine to add variable or constant
                if random.random() < self.build_preference[1]:
                    data = random.choice(self.variable_set)
                    node = Node(data)
                    parent_stack[-1].add_child(node)
                else:
                    data = random.randint(*self.const_range)
                    node = Node(data)
                    parent_stack[-1].add_child(node)
                children_stack[-1] -= 1
                while children_stack[-1] == 0:
                    children_stack.pop()
                    if not children_stack:
                        return None
                    children_stack[-1] -= 1
                    parent_stack.pop()

    def __flatten(self) -> list[Node]:
        node = self.id.children[0]
        nodes = [node]
        if not isinstance(node.data, Function):
            return nodes
        parent_stack = [node]
        children_stack = [node.data.arity - len(node.data.fixed_params)]
        while children_stack:
            parent = parent_stack[-1]
            free_var = parent.data.arity - len(parent.data.fixed_params)
            node = parent.children[free_var - children_stack[-1]]
            nodes.append(node)
            if isinstance(node.data, Function):
                children_stack.append(node.data.arity - len(node.data.fixed_params))
                parent_stack.append(node)
            else:
                if node.is_ts:
                    self.ttl_shift = node.data
                children_stack[-1] -= 1
                while children_stack[-1] == 0:
                    children_stack.pop()
                    if not children_stack:
                        return nodes
                    children_stack[-1] -= 1
                    parent_stack.pop()

    def __get_subnode(
        self,
        ex_terminal: bool = False,
        ex_ts_const: bool = False,
        for_cross: bool = False,
    ) -> Node:
        if ex_terminal:
            # exclude terminal
            candidates = [
                node for node in self.nodes if isinstance(node.data, Function)
            ]
        elif ex_ts_const:
            # exclude time-series node
            candidates = [node for node in self.nodes if not node.is_ts]
        else:
            # no exclusion
            candidates = self.nodes
        if for_cross:
            # according to Koza's (1992) approach of choosing functions with 90% and terminals with 10%
            probs = np.array(
                ([9 if isinstance(node.data, Function) else 1 for node in candidates])
            )
            probs = np.cumsum(probs / probs.sum())
            return candidates[np.searchsorted(probs, random.random())]
        else:
            return random.choice(candidates)

    def execute(self, X: pd.DataFrame) -> Any:
        """
        Execute the program according to X
        @param X: training data
        """
        outcome = np.hstack(
            [np.full(self.ttl_shift, np.nan), self.nodes[0](X)[self.ttl_shift :]]
        )
        if self.transformer is not None:
            outcome = self.transformer(X, outcome, **self.transformer_kwargs)
        return outcome

    def fitness(self, X: pd.DataFrame, benchmark: pd.Series) -> float:
        """
        Evaluate the penalized fitness of the program according to X, benchmark
        @param X: training data
        @param benchmark: first input of metric()
        """
        if self.metric is None:
            raise ValueError("metric must be set")
        raw_fitness = self.metric(benchmark, self.execute(X))
        penalty = self.parsimony_coefficient * len(self) * self.metric.sign
        return raw_fitness - penalty

    def crossover(self, donor: "SyntaxTree") -> "SyntaxTree":
        self_copy, donor_copy = deepcopy(self), deepcopy(donor)
        # it is not good to crossover on time-series constant node, since subtree is always vector but scalar
        own_node = self_copy.__get_subnode(ex_ts_const=True, for_cross=True)
        donor_node = donor_copy.__get_subnode(ex_ts_const=True, for_cross=True)
        if not all([own_node, donor_node]):
            return self_copy
        own_parent = own_node.parent
        own_parent.chg_child(own_node, donor_node)
        # update flatten tree
        self_copy.nodes = self_copy.__flatten()
        return self_copy

    def subtree_mutate(self) -> "SyntaxTree":
        # subtree mutation is basically crossover with a random tree
        donor = SyntaxTree(
            id="anonym",
            init_depth=self.init_depth,
            init_method=self.init_method,
            function_set=self.function_set,
            variable_set=self.variable_set,
            const_range=self.const_range,
            ts_const_range=self.ts_const_range,
            build_preference=self.build_preference,
        )
        return self.crossover(donor)

    def hoist_mutate(self) -> "SyntaxTree":
        self_copy = deepcopy(self)
        if len(self_copy.nodes) == 1:
            # not able to hoist mutate
            return self_copy
        upper_node = self_copy.__get_subnode(ex_terminal=True)
        if upper_node is None:
            # not able to hoist mutate
            return self_copy
        upper_parent = upper_node.parent
        subtree = deepcopy(self)
        subtree.id.chg_child(subtree.nodes[0], upper_node)
        subtree.nodes = subtree.__flatten()
        lower_node = subtree.__get_subnode(ex_ts_const=True)
        if lower_node is None:
            # not able to hoist mutate
            return self_copy
        upper_parent.chg_child(upper_node, lower_node)
        # update flatten tree
        self_copy.nodes = self_copy.__flatten()
        return self_copy

    def point_mutate(self) -> "SyntaxTree":
        self_copy = deepcopy(self)
        mutation_node = self_copy.__get_subnode()
        mutation_parent = mutation_node.parent
        if isinstance(mutation_node.data, Function):
            # function node mutation
            function_candidates = [
                function
                for function in self.function_set
                if function.arity - len(function.fixed_params)
                == mutation_node.data.arity - len(mutation_node.data.fixed_params)
                and function.is_ts == mutation_node.data.is_ts
            ]
            if not function_candidates:
                return self_copy
            replacement = random.choice(function_candidates)
            new_node = Node(replacement)
            for node in mutation_node.children:
                new_node.add_child(node)
            # correct children order
            new_node.children = new_node.children[::-1]
        elif mutation_node.is_ts:
            # time-series constant node mutation
            replacement = random.randint(*self.ts_const_range)
            new_node = Node(replacement, is_ts=True)
        else:
            # variable or constant node mutation
            if random.random() < self.build_preference[1]:
                replacement = random.choice(self.variable_set)
            else:
                replacement = random.randint(*self.const_range)
            new_node = Node(replacement)
        mutation_parent.chg_child(mutation_node, new_node)
        # update flatten tree
        self_copy.nodes = self_copy.__flatten()
        return self_copy

    def reproduce(self) -> "SyntaxTree":
        return deepcopy(self)
