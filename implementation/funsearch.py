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

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any
import traceback
import asyncio
from . import code_manipulation
from . import config as config_lib
from . import evaluator
from . import programs_database
from . import sampler


async def main(specification: str, test_inputs: Sequence[Any], config: config_lib.Config):
    """
    Launches a FunSearch experiment.

    Args:
      specification (str): The specification of the experiment. It's a string representing python code.

      test_inputs (Sequence[Any]): The test inputs for the experiment.
      config (config_lib.Config): The configuration for the experiment. See `config.py` for more details.

    Returns:
      None
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)

    program = code_manipulation.text_to_program(specification)
    output_type = ''
    for func in program.functions:
        if func.name == function_to_run:
            output_type = func.return_type
            print(f'output type is {output_type}')
            break

    database_populated = asyncio.Event()
    database = programs_database.ProgramsDatabase(
        config.programs_database, program, function_to_evolve, database_populated
    )

    queue = asyncio.Queue()
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(
            evaluator.Evaluator(
                database,
                program,
                queue,
                function_to_evolve,
                function_to_run,
                test_inputs,
                config.docker_image,
                output_type=output_type
            )
        )
    # We send the initial implementation to be analysed by one of the evaluators.
    initial = program.get_function(function_to_evolve).body
    await queue.put((initial, None, None))
    llm = sampler.ReplicateLLM(config.samples_per_prompt, queue)
    samplers = [
        sampler.Sampler(
            database, config.total_llm_samples, llm
        )
        for _ in range(config.num_samplers)
    ]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    try:
        producer_tasks = [asyncio.create_task(s.sample()) for s in samplers]
        consumer_tasks = [asyncio.create_task(e.analyse()) for e in evaluators]
        await asyncio.gather(*producer_tasks)
        await queue.join()
        # Enqueue sentinel values
        for _ in range(config.num_evaluators):
            await queue.put(None)

        # Wait for consumers to finish
        await asyncio.gather(*consumer_tasks)
    except Exception as e:
        traceback.print_exc()
        print(e)
    finally:
        await database.report()


def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "run")
    )
    if len(run_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.run`.")
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "evolve")
    )
    if len(evolve_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.evolve`.")
    return evolve_functions[0], run_functions[0]
