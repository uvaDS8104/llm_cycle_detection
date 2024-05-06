# Graph Reasoning with Large Language Models

This project evaluates and enhances the graph reasoning capabilities of large language models (LLMs) for tasks such as cycle detection and clustering coefficient computation. It also includes runtime analysis and robustness checks to assess the scalability and sensitivity of LLMs to phrasing variations.

## File Structure

- `cluster.py`: Python script for evaluating LLMs on the clustering coefficient task.
- `cycle.py`: Python script for evaluating LLMs on the cycle detection task.
- `data/`: Directory containing the datasets used for the experiments.
- `environment.yml`: Conda environment file specifying the dependencies for the project.
- `graph_generator.py`: Python script for generating random graphs with varying sizes and densities.
- `prompt/`: Directory containing prompt templates for different prompting strategies.

## Setup

1. Clone the repository:

git clone git@github.com:uvaDS8104/llm_cycle_detection.git

2. Navigate to the project directory:

cd graph-reasoning-llm

3. Create and activate the Conda environment:

conda env create -f environment.yml
conda activate graph-reasoning-llm

## Usage

To run the experiments, use the following commands:

- Cycle Detection:

python cycle.py --model text-davinci-003 --mode easy --prompt CoT

- Clustering Coefficient:

python cluster.py --model text-davinci-003 --mode easy --prompt CoT

Adjust the arguments according to your desired settings:
- `--model`: Specify the LLM to use (default: text-davinci-003).
- `--mode`: Specify the difficulty level of the graphs (easy, medium, hard).
- `--prompt`: Specify the prompting strategy (CoT, none, PROGRAM, k-shot, Instruct, Algorithm, hard-CoT).

For more advanced usage and additional arguments, refer to the code files `cycle.py` and `cluster.py`.

## Dataset

The `data/` directory contains the datasets used for the experiments. The `graph_generator.py` script can be used to generate random graphs with varying sizes and densities.

## Prompts

The `prompt/` directory contains prompt templates for different prompting strategies used in the experiments.

## References

- Wang, H., et al. (2023). Can language models solve graph problems in natural language?
- Guo, J., et al. (2023). GPT4Graph: Can large language models understand graph structured data?

## License

This project is licensed under the [MIT License](LICENSE).








