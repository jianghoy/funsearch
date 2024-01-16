# Jianghong's Fork of FunSearch

[original repo](https://github.com/google-deepmind/funsearch)
This repository contains Jianghong's implementation of infra of [funsearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), including:
1. language model client for generating new programs.
2. sandbox for executing untrusted code.
3. distributed system for running FunSearch.

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
```
python main.py -s specs_example_cap_set.py -t test_inputs_example_cap_set.txt
```

**TODO: update options**
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
  --programs_database PROGRAMS_DATABASE
                        programs_database (default: <dataclasses._MISSING_TYPE object at 0x101372310>)
  --num_samplers NUM_SAMPLERS
                        num_samplers (default: 15)
  --num_evaluators NUM_EVALUATORS
                        num_evaluators (default: 140)
  --samples_per_prompt SAMPLES_PER_PROMPT
                        samples_per_prompt (default: 4)
  --specification_file SPECIFICATION_FILE, -s SPECIFICATION_FILE
                        Path to the specification file. Specification contains the code to evolve and the code to run, denoted using decorators
                        @funsearch.evolve and @funsearch.run. Decorators in spec files are just for annotation purposes only. In fact they're
                        disabled per code_manpulation.ProgramVisitor See specs_example_cap_set.py for an example.
  --test_inputs_file TEST_INPUTS_FILE, -t TEST_INPUTS_FILE
                        Path to the file containing the test inputs. Each line of the file is a test input, and the test inputs are separated by new
                        lines. See test_inputs_example_cap_set.txt for an example.

### `specification_file`

### `test_input_file`
