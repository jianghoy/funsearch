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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence
from abc import ABC, abstractmethod
import replicate


import numpy as np

from . import evaluator
from . import programs_database


class LLM(ABC):
    """Abstract language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    @abstractmethod
    def _draw_sample(self, prompt: str) -> str:
        pass


# https://replicate.com/meta/codellama-34b-python
CODELLAMA_34B_PYTHON = "meta/codellama-34b-python:e4cb69045bdb604862e80b5dd17ef39c9559ad3533e9fd3bd513cc68ff023656"


class ReplicateLLM(LLM):
    """Concrete implementation of LLM that provides a language model."""

    def __init__(self, samples_per_prompt: int, additional_arg: str) -> None:
        super().__init__(samples_per_prompt)
        self.additional_arg = additional_arg

    def _draw_sample(self, prompt: str) -> str:
        print(prompt)
        raise NotImplementedError("stop here for now")
        output = replicate.run(
            CODELLAMA_34B_PYTHON,
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.75,
                "repeat_penalty": 1.1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        )
        return output


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        total_llm_samples: int,
    ) -> None:
        self._database = database
        self._evaluators = evaluators
        self._llm = ReplicateLLM(samples_per_prompt)
        self._total_llm_samples = total_llm_samples
        self.sample_cnt = 0

    def sample(self):
      """Continuously gets prompts, samples programs, sends them for analysis."""
      while sample.sample_cnt < self._total_llm_samples:
        prompt = self._database.get_prompt()
        samples = self._llm.draw_samples(prompt.code)
        # This loop can be executed in parallel on remote evaluator machines.
        for sample in samples:
          chosen_evaluator = np.random.choice(self._evaluators)
          chosen_evaluator.analyse(
              sample, prompt.island_id, prompt.version_generated)
