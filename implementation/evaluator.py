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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
from typing import Any

import docker
import time

from . import code_manipulation
from . import programs_database
import os

import uuid


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        program: code_manipulation.Program,
        function_to_evolve_name: str,
        function_to_run_name: str,
        test_inputs: Sequence[Any],
        sandbox_docker_image: str,
        timeout_seconds: int = 10,
    ):
        self._database = database
        self._program = program
        self._function_to_evolve_name = function_to_evolve_name
        self._function_to_run_name = function_to_run_name
        self._test_inputs = test_inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = Sandbox(sandbox_docker_image)

    def analyse(
        self,
        sample_function_body: str,
        island_id: int | None,
        version_generated: int | None,
    ) -> None:
        """Compiles the sample into a program and executes it on test inputs."""
        new_function, program = _add_sample_to_program(
            sample_function_body, version_generated, self._program, self._function_to_evolve_name
        )

        scores_per_test = {}
        skip_registering = False
        for test_input in self._test_inputs:
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run_name, test_input, self._timeout_seconds
            )
            if (
                runs_ok
                and not _calls_ancestor(program, self._function_to_evolve_name)
                and test_output is not None
            ):
                if not isinstance(test_output, (int, float)):
                    raise ValueError("@function.run did not return an int/float score.")
                scores_per_test[test_input] = test_output
            else:
                skip_registering = True
        if not skip_registering:
            self._database.register_function(new_function, island_id, scores_per_test)


class Sandbox:
    """Sandbox for executing generated code."""

    def __init__(self, docker_image):
        self._docker_image = docker_image

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input: str,
        timeout_seconds: int,
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded."""
        # get a random id as file name
        current_directory = os.getcwd()
        id = uuid.uuid4().hex
        host_file_name = f"{current_directory}/tmp_{id}.py"
        volume_file_name = f"/tmp/run.py"
        _write_python_file(host_file_name, program, function_to_run, test_input)

        client = docker.from_env()
        container = client.containers.run(
            self._docker_image,
            name=f"funsearch_sandbox_{id}",
            detach=True,
            volumes={
                host_file_name: {
                    "bind": volume_file_name,
                }
            },
            command=f"python {volume_file_name}",
        )

        _wait_for_finish(container, timeout_seconds)
        if container.status != "exited":
            print(
                f"Container {container.short_id} timed out; will kill and does not take result"
            )
            container.remove(force=True)
            os.remove(host_file_name)
            return None, False
        print(f"Containter exited: {container.short_id}")

        response = container.wait()
        if response["StatusCode"] != 0:
            print(
                f"Container {container.short_id} failed with exit code {response['StatusCode']}; does not take result"
            )
            print(f"Container logs: {container.logs().decode('utf-8')}")
            container.remove(force=True)
            os.remove(host_file_name)
            return None, False

        # Read the container logs
        container_logs = container.logs().decode("utf-8").strip()
        container.remove()
        os.remove(host_file_name)
        if "." in container_logs:
            return float(container_logs), True
        else:
            return int(container_logs), True


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extracts the body of the generated function, trimming anything after it."""
    if not generated_code:
        return ""
    code = f"def fake_function_header():\n{generated_code}"
    tree = None
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            code = "\n".join(code.splitlines()[: e.lineno - 1])
    if not code:
        # Nothing could be saved from `generated_code`
        return ""

    visitor = _FunctionLineVisitor("fake_function_header")
    visitor.visit(tree)
    body_lines = code.splitlines()[1 : visitor.function_end_line]
    return "\n".join(body_lines) + "\n\n"


def _add_sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """Rename sampled new function to default name and return program with sampled function."""
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            body, f"{function_to_evolve}_v{version_generated}", function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Returns whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        # In `program` passed into this function the most recently generated
        # function has already been renamed to `function_to_evolve` (wihout the
        # suffix). Therefore any function call starting with `function_to_evolve_v`
        # is a call to an ancestor function.
        if name.startswith(f"{function_to_evolve}_v"):
            return True
    return False


def _wait_for_finish(container, timeout):
    start_time = time.time()
    while time.time() - start_time < timeout:
        container.reload()
        if container.status != "running":
            break
        time.sleep(0.5)
    return time.time() - start_time < timeout


def _write_python_file(file_name: str, code: str, func_to_run: str, test_input: str):
    """Write generated code into a python file. Also add"""
    with open(file_name, "w") as f:
        for line in code.split("\n"):
            if "@funsearch." not in line:
                f.write(line)
                f.write("\n")
        f.write("\n")
        f.write("if __name__ == '__main__':\n")
        f.write(f"  print({func_to_run}({test_input}))")
