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

import textwrap

from absl.testing import parameterized

from implementation import evaluator
import os

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock, call
from implementation.evaluator import Evaluator
from implementation.code_manipulation import Program


class EvaluatorTest(parameterized.TestCase):
    def test_trim_function_body_docstring(self):
        code = '''\
  x = 1

  return 0
"""Docstring"""'''
        desired = """\
  x = 1

  return 0

"""
        actual = evaluator._trim_function_body(code)
        self.assertEqual(desired, actual)

    def test_trim_function_body_function(self):
        code = """\
  return 0
def new_f():"""
        desired = """\
  return 0

"""
        actual = evaluator._trim_function_body(code)
        self.assertEqual(desired, actual)

    def test_trim_function_body_empty(self):
        code = """  return 0\n"""
        desired = """  return 0\n\n"""
        actual = evaluator._trim_function_body(code)
        self.assertEqual(desired, actual)

    def test_trim_function_indentation_corner_case(self):
        code = textwrap.dedent(
            """\
          return (1 +
        2)
        def unfinished_code("""
        )
        desired = textwrap.dedent(
            """\
          return (1 +
        2)

        """
        )
        actual = evaluator._trim_function_body(code)
        self.assertEqual(desired, actual)

    def test_trim_function_backlash_corner_case(self):
        code = textwrap.dedent(
            """\
            return score + ((el[0] + 1) * (el[0] + 2) * el[1] / 6 == el[2])\\
         + ((el[0] + 1) * (el[0] + 2) * (el[0] + 3) * el[1] / 24 == el[2])\\
         + ((el[0] + 1) * (el[0] + 2) * el[1] * el[2] / 6 == n)\\
         + ((el[0] + 1) * (el[0] + 2) * el[1] * el[2] / 3 == n + el[0])\\

        """
        )
        actual = evaluator._trim_function_body(code)
        self.assertEqual(actual, code)

    def test_write_file(self):
        code = textwrap.dedent(
            """\
        def f(input: str) -> int:
          return len(input)
        """
        )
        filename = "test_output.py"
        evaluator._write_python_file(filename, code, "f", '"Hello World!"')
        desired = code + "if __name__ == '__main__':\n  print(f(\"Hello World!\"))"

        with open(filename) as f:
            actual_lines = f.readlines()
            actual_lines = [line.strip() for line in actual_lines if line.strip()]
            desired_lines = desired.splitlines()
            desired_lines = [line.strip() for line in desired_lines if line.strip()]
            self.assertEqual(desired_lines, actual_lines)
        os.remove(filename)

    def test_write_file_remove_annotations(self):
        code = textwrap.dedent(
            """\
        @funsearch.run()
        def f(input: str) -> int:
          return len(input)
        """
        )
        filename = "test_output_2.py"
        evaluator._write_python_file(filename, code, "f", '"Hello World!"')
        # remove first line @funsearch in code
        desired = (
            code[code.find("def f") :]
            + "if __name__ == '__main__':\n  print(f(\"Hello World!\"))"
        )

        with open(filename) as f:
            actual_lines = f.readlines()
            actual_lines = [line.strip() for line in actual_lines if line.strip()]
            desired_lines = desired.splitlines()
            desired_lines = [line.strip() for line in desired_lines if line.strip()]
            self.assertEqual(desired_lines, actual_lines)
        os.remove(filename)


new_function = MagicMock()
new_function.__str__ = lambda x: "new_function"


def sandbox_run_side_effects(test_input: str, program: str):
    if test_input == "1":
        return ("1", 1, True)
    elif test_input == "2":
        return ("2", 2, True)
    elif test_input == "3":
        return ("3", 3, True)
    else:
        raise Exception("Invalid test input")


def add_sample_side_effects(
    code: str, version_generated: int | None, template: Program, func_to_evolve: str
):
    if code == "code0":
        return (new_function, "program0")
    elif code == "code1":
        return (new_function, "program1")
    elif code == "code2":
        return (new_function, "program2")
    else:
        raise Exception("Invalid test input")


class AsyncEvaluatorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.queue = asyncio.Queue()
        # Setup any additional fixtures your consumer needs

    @patch("implementation.evaluator._calls_ancestor")
    @patch("implementation.evaluator.Evaluator._sandbox_run", new_callable=AsyncMock)
    @patch("implementation.evaluator._add_sample_to_program")
    async def test_consumer_processes_items(
        self, mock_add_sample, mock_sandbox_run, mock_calls_ancestor
    ):
        mock_database = MagicMock()
        mock_program = MagicMock()
        func_to_evolve = "evolve"
        func_to_run = "run"
        evaluator = Evaluator(
            mock_database,
            mock_program,
            self.queue,
            func_to_evolve,
            func_to_run,
            ["1", "2", "3"],
            "docker_image",
        )

        # Enqueue test items
        test_item_0 = ("code0", 1, 0)
        test_item_1 = ("code1", 1, 1)
        test_item_2 = ("code2", 1, 2)
        test_items = [test_item_0, test_item_1, test_item_2]
        for item in test_items:
            await self.queue.put(item)

        mock_calls_ancestor.return_value = False
        mock_add_sample.side_effect = add_sample_side_effects
        mock_sandbox_run.side_effect = sandbox_run_side_effects
        # Start the consumer
        consumer_task = asyncio.create_task(evaluator.analyse())

        # Add a sentinel value to stop the consumer or implement another stopping mechanism
        await self.queue.put(None)  # Assuming your consumer stops on None

        # Wait for the consumer to process items
        await consumer_task

        self.assertEqual(mock_calls_ancestor.call_count, 9)
        self.assertEqual(mock_add_sample.call_count, 3)
        self.assertEqual(mock_sandbox_run.call_count, 9)
        self.assertEqual(mock_database.register_function.call_count, 3)
        grouped_calls = [
            [call("program0", func_to_evolve) for _ in range(3)],
            [call("program1", func_to_evolve) for _ in range(3)],
            [call("program2", func_to_evolve) for _ in range(3)],
        ]
        mock_calls_ancestor.assert_has_calls(
            [c for grouped_call in grouped_calls for c in grouped_call]
        )

        mock_sandbox_run.assert_has_calls(
            [
                call("1", "program0"),
                call("2", "program0"),
                call("3", "program0"),
                call("1", "program1"),
                call("2", "program1"),
                call("3", "program1"),
                call("1", "program2"),
                call("2", "program2"),
                call("3", "program2"),
            ]
        )
        scores_per_test = {"1": 1, "2": 2, "3": 3}
        mock_database.register_function.assert_has_calls(
            [call(new_function, 1, scores_per_test) for _ in range(3)]
        )


if __name__ == "__main__":
    unittest.main()
