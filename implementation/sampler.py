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
from collections.abc import Sequence
from abc import ABC, abstractmethod
import replicate
import copy
import asyncio

from . import evaluator
from . import programs_database


class LLM(ABC):
    """Abstract language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int, queue: asyncio.Queue) -> None:
        self._samples_per_prompt = samples_per_prompt
        self._queue = queue

    # TODO: make it async
    async def draw_samples_and_send_to_queue(
        self, prompt: programs_database.Prompt
    ) -> None:
        tasks = [
            asyncio.create_task(self._draw_sample_and_send_to_queue(prompt))
            for _ in range(self._samples_per_prompt)
        ]
        await asyncio.gather(*tasks)

    async def _draw_sample_and_send_to_queue(
        self, prompt: programs_database.Prompt
    ) -> None:
        sample = await self._draw_sample(prompt.code)
        await self._queue.put((sample, prompt.island_id, prompt.version_generated))

    @abstractmethod
    async def _draw_sample(self, prompt: str) -> str:
        pass


# https://replicate.com/meta/codellama-34b-python
CODELLAMA_34B_PYTHON = "meta/codellama-34b-python:e4cb69045bdb604862e80b5dd17ef39c9559ad3533e9fd3bd513cc68ff023656"
DEFAULT_INPUT_ARGS = {
    "top_k": 50,
    "top_p": 0.9,
    "max_tokens": 500,
    "temperature": 0.75,
    "repeat_penalty": 1.1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}


class ReplicateLLM(LLM):
    """Concrete implementation of LLM that provides a language model."""

    def __init__(
        self,
        samples_per_prompt: int,
        queue: asyncio.Queue,
        input_args=DEFAULT_INPUT_ARGS,
    ) -> None:
        super().__init__(samples_per_prompt, queue)
        self._input_args = input_args

    # TODO: figure out how to run asyncio here.
    async def _draw_sample(self, prompt: programs_database.Prompt, queue) -> None:
        input = copy.deepcopy(self._input_args)
        input["prompt"] = prompt.code
        output_generator = await replicate.async_run(CODELLAMA_34B_PYTHON, input=input)
        output = "".join(output_generator)
        return output


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    # TODO: samples_per_prompt was original param for llm sampler, one can try tidy it up and remove it.
    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        total_llm_samples: int,
        llm: LLM
    ) -> None:
        self._database = database
        self._llm = llm
        self._total_llm_samples = total_llm_samples
        self.sample_cnt = 0

    async def sample(self):
        while self.sample_cnt < self._total_llm_samples:
            self.sample_cnt += 1
            prompt = self._database.get_prompt()
            await self._llm.draw_samples_and_send_to_queue(prompt)
