from asyncio import Queue
from unittest.mock import MagicMock, AsyncMock
import asyncio
from implementation.sampler import Sampler, LLM, ReplicateLLM
from implementation.programs_database import Prompt
import unittest
from dotenv import load_dotenv

TEST_REPLICATE_INPUT_ARGS = {
    "top_k": 50,
    "top_p": 0.9,
    "max_tokens": 10,
    "temperature": 0.75,
    "repeat_penalty": 1.1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}


class TestLLM(LLM):
    def __init__(self, samples_per_prompt: int, queue: asyncio.Queue) -> None:
        super().__init__(samples_per_prompt, queue)

    async def _draw_sample(self, prompt: str) -> None:
        await asyncio.sleep(0.01)
        return prompt


class TestSampler(unittest.IsolatedAsyncioTestCase):
    async def test_sample(self):
        mock_database = MagicMock()
        total_llm_samples = 2
        samples_per_prompt = 5
        mock_queue = MagicMock()
        mock_queue.put = AsyncMock()
        sampler = Sampler(
            mock_database, total_llm_samples, TestLLM(samples_per_prompt, mock_queue)
        )
        prompt = Prompt("code", 0, 0)
        mock_database.get_prompt.return_value = prompt

        await sampler.sample()

        self.assertEqual(mock_database.get_prompt.call_count, total_llm_samples)
        self.assertEqual(
            mock_queue.put.call_count, total_llm_samples * samples_per_prompt
        )
        mock_queue.put.assert_called_with((prompt.code, prompt.island_id, prompt.version_generated))

    # Comment out because the test is basically testing connection to replicate, and is very slow.
    # async def test_replicate_llm(self):
    #     load_dotenv()
    #     samples_per_prompt = 1
    #     replicate_llm = ReplicateLLM(samples_per_prompt, TEST_REPLICATE_INPUT_ARGS)
    #     mock_queue = MagicMock()
    #     await replicate_llm.draw_samples_and_send_to_queue('prompt', mock_queue)
    #     self.assertEqual(mock_queue.put.call_count, samples_per_prompt)


if __name__ == "__main__":
    unittest.main()
