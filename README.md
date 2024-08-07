# SUBER: Simulated User Behavior Environment for Recommender Systems Applied to the MIND Dataset

<p align="left"> 

</p>

This is a fork of the [original SUBER project](https://github.com/SUBER-Team/SUBER) [[1]] repository to work with the [MIND dataset](https://msnews.github.io/). the original paper is an "[An LLM-based Recommender System Environment](https://arxiv.org/pdf/2406.01631)" by Nathan Coreco*, Giorgio Piatti*, Luca A. Lanzendörfer, Flint Xiaofeng Fan, Roger Wattenhofer.

### Docker Container Development
The development environment has been containerized, the docker/[Dockerfile](docker/Dockerfile) can be built with `docker build -t subercon `. the `docker` folder also contains a copy of the [requirements.txt](docker/requirements.txt) also located at the base of the project.  When building the container it was easier to place it here and just build the container.  Since I'm developing in Windows, [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and configured to work with WSL2. Be sure to search for how to do this and get it done ---- IF developing on Windows. ;-) Otherwise install what's needed for whatever Linux OS you are using.

### Developing with VSCode
VSCode is the standard for this project.  There is a [devcontainer.json](.devcontainer/devcontainer.json) located in a `.devcontainer`.  This is the setup I used to integrated with VSCode.  I developed on Ubuntu Linux which includes wroking on WSL2; while not ideal, it is convenient for creating and debugging code.  When conducting experiments it is best to have a dedicated GPU or preferrably GPUS with at least 24G of GPU memory or more.  The more the merrier. 

There is also a `.vscode` directory containing a file called [launch.json](.vscode/launch.json) which enabled me to use the VSCode integrated debugger while developing.  Very Handy!  

There is a `.github` folder with a [dependabot](.github/dependabot.yml) in it; read about this at GitHub.

After building the subercon container, installing VSCode, be sure to install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). With the `SUBER` code in the workspace, press F1 and search for `Dev Container: Open Folder in Container` and select it.  You'll be prompted for the folder, type in the location of the SUBER folder like `/home/user/SUBER`. This will kick things off and you'll be developing in no time. 


To launch it and display help. Please pay attention to the `CF_train_A2C2`. `CF_tran_A2C` was the original but was heavily modified with an adaptive scheduler, the small MIND dataset and other changes.  

```.bash

python3 -m algorithms.mind.CF_train_A2C2 --help

usage: CF_train_A2C.py [-h]
                       [--llm-model {TheBloke/Llama-2-7b-Chat-GPTQ,TheBloke/Llama-2-13B-chat-GPTQ,TheBloke/vicuna-7B-v1.3-GPTQ,TheBloke/vicuna-13b-v1.3.0-GPTQ,TheBloke/vicuna-33B-GPTQ,TheBloke/vicuna-7B-v1.5-GPTQ,TheBloke/vicuna-13B-v1.5-GPTQ,gpt-3.5-turbo-0613,gpt-4-0613,gpt-4o,TheBloke/Mistral-7B-Instruct-v0.2-GPTQ}]
                       [--llm-rater {2Shot_system_our,1Shot_system_our,0Shot_system_our,0Shot_system_our_1_10,1Shot_system_our_1_10,2Shot_system_our_1_10,2Shot_system_our_one_ten,1Shot_system_our_one_ten,2Shot_invert_system_our,1Shot_invert_system_our}]
                       [--items-retrieval {last_3,most_similar_3_title,most_similar_3_abstract,none,simple_3}] [--user-dataset {mind}] [--perturbator {none,gaussian,greedy}]
                       [--reward-shaping {identity,exp_decay_time,random_watch,same_film_terminate}] [--seed SEED] [--model-device MODEL_DEVICE] [--gamma GAMMA] [--embedding-dim EMBEDDING_DIM]
                       [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --llm-model {TheBloke/Llama-2-7b-Chat-GPTQ,TheBloke/Llama-2-13B-chat-GPTQ,TheBloke/vicuna-7B-v1.3-GPTQ,TheBloke/vicuna-13b-v1.3.0-GPTQ,TheBloke/vicuna-33B-GPTQ,TheBloke/vicuna-7B-v1.5-GPTQ,TheBloke/vicuna-13B-v1.5-GPTQ,gpt-3.5-turbo-0613,gpt-4-0613,gpt-4o,TheBloke/Mistral-7B-Instruct-v0.2-GPTQ}
  --llm-rater {2Shot_system_our,1Shot_system_our,0Shot_system_our,0Shot_system_our_1_10,1Shot_system_our_1_10,2Shot_system_our_1_10,2Shot_system_our_one_ten,1Shot_system_our_one_ten,2Shot_invert_system_our,1Shot_invert_system_our}
  --items-retrieval {last_3,most_similar_3_title,most_similar_3_abstract,none,simple_3}
  --user-dataset {mind}
  --perturbator {none,gaussian,greedy}
  --reward-shaping {identity,exp_decay_time,random_watch,same_film_terminate}
  --seed SEED
  --model-device MODEL_DEVICE
  --gamma GAMMA
  --embedding-dim EMBEDDING_DIM
  --learning_rate LEARNING_RATE

```

Note: The learning rate is now linearly descending.  By that I mean that it will make itself smaller as learning progresses.  Please don't had a learning_rate argument -- this will be fied in the future. 

Here is an example a full command line run.


```.bash
python3 -m algorithms.mind.CF_train_A2C2 --llm-model=TheBloke/Llama-2-13B-chat-GPTQ --llm-rater=2Shot_system_our --perturbator=gaussian --items-retrieval=most_similar_3_title --reward-shaping=exp_decay_time --embedding-dim=512 --gamma=0.95 --seed=42

```

The above will run with the model and other parameters specified.


This repository accompanies our research paper titled "**An LLM-based Recommender System Environment**".

#### Our paper:

"**[An LLM-based Recommender System Environment](http://arxiv.org/abs/2406.01631)**" by *Nathan Coreco\*, Giorgio Piatti\*, Luca A. Lanzendörfer, Flint Xiaofeng Fan, Roger Wattenhofer*.

**Citation:**

```bibTeX
@misc{corecco2024llmbased,
      title={An LLM-based Recommender System Environment}, 
      author={Nathan Corecco and Giorgio Piatti and Luca A. Lanzendörfer and Flint Xiaofeng Fan and Roger Wattenhofer},
      year={2024},
      eprint={2406.01631},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

We open source `SUBER`, a Reinforcement Learning (RL) environment for recommender systems.
We adopt the standard of [Farama's Gymnasium](https://gymnasium.farama.org/) and are compatible with  [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). `SUBER` is a framework for simulating user behavior in recommender systems, for which we provide a set of recommendation environments: movies, and books.

Along with `SUBER`, we provide instructions on how to run the code.

## Guide

### Requirements

Python 3.9 and CUDA 11.8 with PyTorch 2. Run [setup.sh](setup.sh) for a quick setup using conda, or see [requirements.txt](requirements.txt) for the full list of Python packages.

We only tested the code on Linux. We suggest to use a virtual conda environment to install the required packages.




### General environment arguments
The following arguments are available for all environments, and can be passed to ablations and RL training:

```
--llm-model: LLM model to use for the environment. We support Llama based models via exllama GPTQ and OpenAI GPT-3.5 and GPT-4.
--llm-rater: Prompting strategy to query the LLM
--items-retrieval: Items retrieval strategy
--perturbator: Reward perturbator strategy 
--reward-shaping: Reward shaping strategy
--user-dataset: Dataset of users to use
```


### Run ablations
Run the following command to run a config of ablations for the movie environment:
``` 
python3 -m ablations.movies.run <args>
```

Run the following command to run a config of ablations for the book environment:
```
python3 -m ablations.books.run  <args>
```

### Run RL training
Run the following command to run the RL training for the movie environment:
```
python3 -m algorithms.movies.CF_train_A2C
```
Additional arguments are available, see `algorithms/movies/CF_train_A2C.py` file for more details.


## Developer Notes
We provide two main recommendation environments: movies and books. Each environment has its own folder in the `environment` folder.
Each environment has its own `configs.py` file, which contains the default configuration and arguments for the environment. 

### Implementing a new environment
To implement a new environment, you need to create a new folder in the `environment` folder, and implement the following classes:
- `Item`: extends the `Item` class in `items.py`
- `User`: extends the `User` class in `users.py`
- `Rater`: extends the `Rater` class in `rater.py`
- `Loader`: extends the `Loader` class in `loader.py`
and the following files:
- `configs.py`: default configuration and arguments for the environment

We refer to the `movies` and `books` environments for examples.

### Project structure

```
ablations/
    movies/
        datasets/                   -- Ablation datasets
        reports/                    -- Results folder (output)
        src/                        -- Source files for the ablatiosn test cases
        run.py                      -- Run Book ablations (*)
        run_gpt.py                  -- Run Book ablations - less sampling (*)
        run_sampling_analysis.sh    -- Run sampling analysis for all reports folders
    movies/
        datasets/                   -- Ablation datasets
        reports/                    -- Results folder  (output)
        src/                        -- Source files for the ablatiosn test cases
        run.py                      -- Run Movie ablations (*)
        run_gpt.py                  -- Run Movie ablations - less sampling (*)
        run_sampling_analysis.sh    -- Run sampling analysis for all reports folders

algorithms/                         -- RL train code
    movies/                         -- RL trainining and analysis code
    wrappers.py                     -- Gymnasium wrappers to use Stable Baselines-3
environment/
    LLM/                            -- LLM model specific subfolders
        guidance/                   -- Guidance model class, used for user generator
        exllama.py                  -- ExLLAMA model class (for GPTQ inference)
        llm.py                      -- Abstract LLM model class, used by the rater
        rater.py                    -- Rater class (base class), used to rate items based on user characteristics and items features, 
                                       all prompting strategy need to extend this class
    books/                          -- Book specific environment components
        datasets/                   -- TMDB data saved in JSON format
        rater_prompts/              -- Rater prompts for movies environment
        users_generator/            -- Users generator for movies environment
        book.py                     -- Book class for features
        book_loader.py              -- Load movies from CSV dataset to Book class
        configs.py                  -- Default configuration and arguments for the movie environment
    
    movies/                         -- Movie specific environment components
        datasets/                   -- TMDB data saved in JSON format
        rater_prompts/              -- Rater prompts for movies environment
        users_generator/            -- Users generator for movies environment
        movie.py                    -- Movie class for features
        movies_loader.py            -- Load movies from JSON dataset to Movie class
        configs.py                  -- Default configuration and arguments for the movie environment
    users/                          -- 
        datasets/                   -- Dataset sampling during users generation
        user.py                     -- User class
        user_loader.py              -- UserLoaders support CSV and list of User objects
    env.py                          -- Gymnasium environment
    items.py                        -- Abstract class for item, all environment need to extend this class
    memory.py                       -- Memory for each user containing item_id and rating for past interacions
    items_perturbation.py           -- Perturbation components
    items_retrival.py               -- Retrieval components
    reward_perturbator.py           -- Reward perturbation components
    reward_shaping.py               -- Reward shaping components   
```

## License

The `SUBER` code is released under the CC BY 4.0 license. For more information, see [LICENSE](LICENSE).
