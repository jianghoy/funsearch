# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from datetime import datetime
import os
from typing import Any

from absl import logging
import numpy as np
import scipy

from . import code_manipulation
from . import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """

    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._report_root_dir: str = config.report_dir

        # Initialize empty islands.
        self._islands: list[Island] = []
        for i in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    function_to_evolve,
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                    i
                )
            )

        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def register_function(
        self,
        function: code_manipulation.Function,
        island_id: int | None,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers `function` in the database. The function, is the function we wish to involve, denoted by @funsearch.evolve"""
        # TODO: In an asynchronous implementation we should consider the possibility of
        # registering a function on an island that had been reset after the prompt
        # was generated. 
        # Basically, you need to assign a island version to each island, and each time
        # you reset the island, the island's version goes up. When you sample from an island, you need to keep
        # the island version, and when you add back you check and see if island version is bigger than the island
        # version you get when you sample the program. If yes, then you cannot add it back.
        #
        # However, you also need to factor in concurrency.
        if island_id is None:
            # This is a function added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_function_in_island(function, island_id, scores_per_test)
        else:
            self._register_function_in_island(function, island_id, scores_per_test)

        # Check whether it is time to reset an island.
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def _register_function_in_island(
        self,
        function: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers `function` in the specified island."""
        self._islands[island_id].register_function(function, scores_per_test)

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.

        best_score_per_island = np.array(
            [self._islands[i]._best_score for i in range(len(self._islands))]
        )
        indices_sorted_by_score: np.ndarray = np.argsort(
            best_score_per_island
            + np.random.randn(len(best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
                island_id
            )
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._islands[founder_island_id]._best_function
            founder_scores = self._islands[founder_island_id]._best_scores_per_test
            self._register_function_in_island(founder, island_id, founder_scores)

    def report(self) -> None:
        # if self.report_dir not exist create new directory
        if not os.path.exists(self._report_root_dir):
            os.makedirs(self._report_root_dir)
        # make directory under self.report_dir with naming as report_datetime
        report_dir = (
            self._report_root_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(report_dir)
        # write the report to a file
        with open(report_dir + "/report.txt", "w") as f:
            f.write(f"Best score per islands:\n")
            # order by key of best_score_per_test,and best_score_per_test to file
            for island in self._islands:
                best_score_per_test = island._best_scores_per_test
                sorted_keys = sorted(best_score_per_test.keys())
                f.write("{")
                for key in sorted_keys:
                    f.write(f"{key}: {best_score_per_test[key]},")
                f.write("}\n")
        for i in range(len(self._islands)):
            with open(report_dir + f"/island_{i}_best_program.py", "w") as f:
                # TODO: update this to get best program instead of function
                f.write(str(self._islands[i]._best_function))


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
        id: int
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._clusters: dict[Signature, Cluster] = {}
        self._num_functions: int = 0

        self._best_score = -float("inf")
        self._best_function = None
        self._best_scores_per_test = None
        self._id = id

    def register_function(
        self,
        function: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a function on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)
        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, function)
        else:
            self._clusters[signature].register_function(function)
        self._num_functions += 1

        if score > self._best_score:
            self._best_score = score
            self._best_function = function
            self._best_scores_per_test = scores_per_test
            logging.info("Best score of island %d increased to %s", self._id, score)


    # TODO: test this.
    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures]
        )

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
            1 - (self._num_functions % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # functions into the prompt as example.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities
        )
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_function())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
        self, implementations: Sequence[code_manipulation.Function]
    ) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f"{self._function_to_evolve}_v{i}"
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                )
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name
            )
            versioned_functions.append(
                code_manipulation.text_to_function(implementation)
            )

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body="",
            docstring=(
                "Improved version of "
                f"`{self._function_to_evolve}_v{next_version - 1}`."
            ),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of functions on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._functions: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_function(self, function: code_manipulation.Function) -> None:
        """Adds `function` to the cluster."""
        self._functions.append(function)
        self._lengths.append(len(str(function)))

    def sample_function(self) -> code_manipulation.Function:
        """Samples a function, giving higher probability to shorther functions."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._functions, p=probabilities)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


# I don't think this is correct. Because scores_per_test is a map and the output will not be mapped to a
# correct value ot a specific key, and comparing with wrong key will yield bad result.
def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score."""
    return scores_per_test[max(scores_per_test.keys())]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """
    Represents test scores as a canonical signature.
    We define the signature of a evolution target function as the tuple containing the evolution target function's
    scores on each of the inputs (e.g., the cap set size for each input n).
    """
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))
