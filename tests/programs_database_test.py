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

import copy

from absl.testing import parameterized
import numpy as np

from implementation import code_manipulation
from implementation import config
from implementation import programs_database
import unittest
from unittest.mock import MagicMock
_SKELETON = '''
"""Finds large cap sets."""
import numpy as np
import utils_capset


def evaluate(n: int) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  capset = solve(n)
  return len(capset)


def priority(element, n):
  """Returns the priority with which we want to add `element` to the cap set."""
  return 0.0
'''
_EXPECTED_INITIAL_PROMPT = '''
"""Finds large cap sets."""
import numpy as np
import utils_capset


def priority_v0(element, n):
  """Returns the priority with which we want to add `element` to the cap set."""
  return 0.0


def priority_v1(element, n):
  """Improved version of `priority_v0`."""

'''

_SAMPLE_A = """\
  priority = element
  #######
  # Code from lowest-scoring sampled program.
  #######
  return ...\
"""
_SAMPLE_B = """\
  priority = element ** 2
  #######
  # Code from highest-scoring sampled program.
  #######
  return ...\
"""

# A prompt like this appears in the Extended Data of the paper.
_EXPECTED_PROMPT = '''
"""Finds large cap sets."""
import numpy as np
import utils_capset


def priority_v0(element, n):
  """Returns the priority with which we want to add `element` to the cap set."""
  priority = element
  #######
  # Code from lowest-scoring sampled program.
  #######
  return ...


def priority_v1(element, n):
  """Improved version of `priority_v0`."""
  priority = element ** 2
  #######
  # Code from highest-scoring sampled program.
  #######
  return ...


def priority_v2(element, n):
  """Improved version of `priority_v1`."""

'''


class AsyncProgramsDatabaseTest(unittest.IsolatedAsyncioTestCase):
    async def test_initial_prompt(self):
        """Verifies that the first prompt looks as expected."""

        # Create a programs database.
        template = code_manipulation.text_to_program(_SKELETON)
        function_to_evolve = "priority"
        database = programs_database.ProgramsDatabase(
            config=config.ProgramsDatabaseConfig(functions_per_prompt=5),
            template=template,
            function_to_evolve=function_to_evolve,
            populated=MagicMock()
        )
        # Register the initial implementation provided in the skeleton template.
        await database.register_function(
            function=template.get_function(function_to_evolve),
            island_id=None,
            scores_per_test={"unused": -1},
        )
        prompt = await database.get_prompt()
        # Verify the first prompt.
        self.assertEqual(prompt.code, _EXPECTED_INITIAL_PROMPT)

    async def test_destroy_islands(self):
        template = code_manipulation.text_to_program(_SKELETON)
        function_to_evolve = "priority"
        database = programs_database.ProgramsDatabase(
            config=config.ProgramsDatabaseConfig(num_islands=10),
            template=template,
            function_to_evolve=function_to_evolve,
            populated=MagicMock()
        )
        scores = [7, 3, 5, 6, 7, -2, 0, -1, 4, 3]
        unused_function = template.get_function(function_to_evolve)
        for i, score in enumerate(scores):
            await database.register_function(
                function=unused_function,
                island_id=i,
                scores_per_test={"unused_input": score},
            )
        await database.register_function(
            unused_function, island_id=7, scores_per_test={"unused_input": 17}
        )

        expected_scores = list(scores)
        expected_scores[7] = 17
        best_score_per_island = [island._best_score for island in database._islands]
        self.assertSequenceEqual(best_score_per_island, expected_scores)

        np.random.seed(0)
        await database.reset_islands()
        expected_kept = set([0, 2, 3, 4, 7])
        min_kept = min(expected_scores[i] for i in expected_kept)
        for i, score in enumerate(expected_scores):
            if i in expected_kept:
                self.assertEqual(database._islands[i]._best_score, score)
            else:
                self.assertGreaterEqual(database._islands[i]._best_score, min_kept)


class ProgramsDatabaseTest(parameterized.TestCase):

    def test_generate_prompt(self):
        """Tests that we build the prompt shown in the paper."""
        template = code_manipulation.text_to_program(_SKELETON)
        function_to_evolve = "priority"
        island = programs_database.Island(
            template=template,
            function_to_evolve=function_to_evolve,
            functions_per_prompt=2,
            cluster_sampling_temperature_init=1.0,
            cluster_sampling_temperature_period=30_000,
            id=1,
        )
        sample_a = copy.deepcopy(template.get_function(function_to_evolve))
        sample_a.body = _SAMPLE_A
        sample_b = copy.deepcopy(template.get_function(function_to_evolve))
        sample_b.body = _SAMPLE_B
        prompt = island._generate_prompt([sample_a, sample_b])
        self.assertEqual(prompt, _EXPECTED_PROMPT)

    @parameterized.parameters(
        [
            dict(logits=np.array([10, 9, -1000], dtype=np.float32)),
            dict(logits=np.array([10, 9, -1000], dtype=np.int32)),
            dict(logits=np.zeros(50)),
        ]
    )
    def test_softmax(self, logits: np.ndarray):
        probs_lower_temp = programs_database._softmax(logits, temperature=1.0)
        self.assertAlmostEqual(np.sum(probs_lower_temp), 1.0, places=6)
        self.assertTrue(np.all(probs_lower_temp >= 0))
        self.assertTrue(np.all(probs_lower_temp <= 1))

        probs_higher_temp = programs_database._softmax(logits, temperature=5.0)
        self.assertAlmostEqual(np.sum(probs_higher_temp), 1.0, places=6)
        self.assertTrue(np.all(probs_higher_temp >= 0))
        self.assertTrue(np.all(probs_higher_temp <= 1))

        if not np.all(logits == logits[0]):
            # The lower the temperature, the more confident we are on our top choice.
            self.assertGreater(np.max(probs_lower_temp), np.max(probs_higher_temp))

    @parameterized.parameters(
        [
            dict(temperature=1, expected=[0.6590012, 0.242433, 0.0985659]),
            dict(temperature=2, expected=[0.50168777, 0.304289, 0.19402324]),
        ]
    )
    def test_softmax_output(self, temperature, expected):
        logits = np.array([2.0, 1.0, 0.1])
        probabilities = programs_database._softmax(logits, temperature)
        np.testing.assert_array_almost_equal(
            probabilities, np.array(expected), decimal=5
        )

    @parameterized.parameters(
        [
            dict(logits=np.array([100, 200, 300, np.nan])),
            dict(logits=np.array([100, np.inf, 300, 200])),
            dict(logits=np.array([-np.inf, 200, 300, 100])),
        ]
    )
    def test_softmax_non_finite_error(self, logits):
        with self.assertRaisesRegex(
            ValueError,
            r"`logits` contains non-finite value\(s\)",
        ):
            programs_database._softmax(logits, temperature=1.0)

    def test__reduce_score(self):
        scores_per_test = {"3": 8, "4": 20}
        self.assertEqual(20, programs_database._reduce_score(scores_per_test))


if __name__ == "__main__":
    unittest.main()
