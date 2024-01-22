# Jianghong's Fork of FunSearch

[original repo](https://github.com/google-deepmind/funsearch)
This repository contains Jianghong's implementation of infra of [funsearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), including:
- [ ] language model client for generating new programs.  
  - [ ] Feature parity: use [Google Cloud Vertex AI Code model](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview)
  - [ ] Feature parity: use [StarCoder](https://github.com/bigcode-project/starcoder) from [vendor ???]()
  - [ ] GPT4
  - [x] [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) from [vendor replicate](https://replicate.com/meta/codellama-34b-python)
- [x] sandbox for executing untrusted code.
- [x] single host `asyncio` based multitasking for running FunSearch efficiently. See below for analysis.

## Why asyncio?
Based on my observation, implementing multi-host or multi-process for funsearch is not beneficial under a certain scale limit. Here I assume a typical user uses a cloud based LLM provider via API.(based on the paper it seems the original experiment infra includes a set of ML accelerators (*The distributed setting enables running many evaluator nodes on inexpensive CPU hardware, while few samplers run on machines with accelerators for fast LLM inference, page 6 of funsearch paper*), which by default likely has to be distributed anyway for model like Palm2). Once we move ML accelerators out of the infra setup, the bottleneck becomes network latency (and cost) of each LLM API call, which is around 1 second. The python method to be executed in sandbox doesn't seem to use a lot of resources either. So the typical workload fits nicely to the `asyncio` model. Given enough budget, one may be able to call hundreds of LLM APIs in parallel, and multi-host can be beneficial in that setting.  

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

### `specification_file`
See [specs_example_cap_set.py](specs_example_cap_set.py) for an example. Basically, you want to add annotation `@funsearch.evolve` to your code to denote which part of the code should be evolved and `@funsearch.run` which part should be evaluation entrypoint.
### `test_input_file`
Test cases input, each line is 1 test case. See [test_inputs_example_cap_set.txt](test_inputs_example_cap_set.txt) for an example.
