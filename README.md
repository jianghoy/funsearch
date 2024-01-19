# Jianghong's Fork of FunSearch

[original repo](https://github.com/google-deepmind/funsearch)
This repository contains Jianghong's implementation of infra of [funsearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), including:
- [ ] language model client for generating new programs.  
  - [ ] Feature parity: use [Google Cloud Vertex AI Code model](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview)
  - [ ] Feature parity: use [StarCoder](https://github.com/bigcode-project/starcoder) from [vendor ???]()
  - [ ] GPT4
  - [ ] [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) from [vendor replicate]()
- [x] sandbox for executing untrusted code.
- [ ] single host thread based distributed system for running FunSearch.

## Installation
### Using poetry
1. Install poetry
2. Run: 
```
poetry install
```
### Manually
Or, you can find all needed dependencies under `[tool.poetry.dependencies]` in `pyproject.toml`
## Usage
### Prepare docker image for testing
The sandbox uses docker to run tests against funsearch generated programs. Sample [Dockerfile](./Dockerfile) for cap_set is provided.
### Add API keys to `.env`
To use remote APIs for generating new python functions, you need to first rename `.env_example` to `.env`, and then add your API key to corresponding entries.
### Run `main.py`
```
python main.py -s specs_example_cap_set.py -t test_inputs_example_cap_set.txt
```
```bash
options:
  -h, --help            show this help message and exit
  --functions_per_prompt FUNCTIONS_PER_PROMPT
                        functions_per_prompt (default: 2)
  --num_islands NUM_ISLANDS
                        num_islands (default: 10)
  --reset_period RESET_PERIOD
                        reset_period (default: 14400)
  --cluster_sampling_temperature_init CLUSTER_SAMPLING_TEMPERATURE_INIT
                        cluster_sampling_temperature_init (default: 0.1)
  --cluster_sampling_temperature_period CLUSTER_SAMPLING_TEMPERATURE_PERIOD
                        cluster_sampling_temperature_period (default: 30000)
  --num_samplers NUM_SAMPLERS
                        num_samplers (default: 15)
  --num_evaluators NUM_EVALUATORS
                        num_evaluators (default: 140)
  --samples_per_prompt SAMPLES_PER_PROMPT
                        samples_per_prompt (default: 4)
  --specification_file SPECIFICATION_FILE, -s SPECIFICATION_FILE
                        Path to the specification file. Specification contains the code to evolve and the code to run,
                        denoted using decorators @funsearch.evolve and @funsearch.run. Decorators in spec files are just for
                        annotation purposes only. In fact they're disabled per code_manpulation.ProgramVisitor See
                        specs_example_cap_set.py for an example.
  --test_inputs_file TEST_INPUTS_FILE, -t TEST_INPUTS_FILE
                        Path to the file containing the test inputs. Each line of the file is a test input, and the test
                        inputs are separated by new lines. See test_inputs_example_cap_set.txt for an example.
```
### `specification_file`
See [specs_example_cap_set.py](specs_example_cap_set.py) for an example. Basically, you want to add annotation `@funsearch.evolve` to your code to denote which part of the code should be evolved and `@funsearch.run`which part should be run.

### `test_input_file`
Test cases input, each line is 1 test case. See [test_inputs_example_cap_set.txt](test_inputs_example_cap_set.txt) for an example.