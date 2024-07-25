import os
import torch
from stable_baselines3 import A2C
import numpy as np
from environment import load_LLM
from environment.mind.configs import get_enviroment_from_args
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
import glob
from huggingface_hub import snapshot_download
import argparse

base_path = os.path.expanduser('~/efs/home/sheller')
from algorithms.logging_config import get_logger
logger = get_logger("suber_logger")

def get_available_models():
    model_dirs = [d for d in os.listdir(base_path) if d.startswith('TheBloke') and os.path.isdir(os.path.join(base_path, d))]
    dirs = []
    models = []
    for dir_name in model_dirs:
        dirs.append(dir_name)
        components = dir_name.split('___')
        keys = ["llm_model", "llm_rater", "items_retrieval", "user_dataset", "news_dataset",
                "perturbator", "reward_shaping", "seed", "model_device", "gamma", "embedding_dim", "learning_rate"]
        model = {}
        model["dir_name"] = dir_name
        model["llm_model"] = components[0]
        parts = model["llm_model"].split('_')
        if len(parts) == 4:
            parts[0] = parts[0] + '/'
            model["llm_model"] = parts[0] + parts[1] + '.' + parts[2] + '.' + parts[3]
        elif len(parts) == 3:
            parts[0] = parts[0] + '/'
            model["llm_model"] = parts[0] + parts[1] + '.' + parts[2]
        elif len(parts) == 2:
            model["llm_model"] = parts[0] + '/' + parts[1]
        model["llm_rater"] = components[1]
        model["items_retrieval"] = components[2]
        model["user_dataset"] = components[3]
        model["news_dataset"] = components[4]
        model["perturbator"] = components[5]
        model["reward_shaping"] = components[6]
        model["seed"] = int(components[7])
        model["model_device"] = components[8].replace('_', ':')
        model["gamma"] = float(components[9].replace('_', '.'))
        model["embedding_dim"] = int(components[10])
        model["learning_rate"] = float(components[11].replace('_', '.'))
        models.append(model)
    return models

def initialize_exllama(model_name, revision):
    model_directory = snapshot_download(repo_id=model_name, revision=revision)
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    config = ExLlamaConfig(model_config_path)
    config.max_seq_len = 4096
    config.model_path = model_path

    model = ExLlama(config)
    tokenizer = ExLlamaTokenizer(tokenizer_path)
    cache = ExLlamaCache(model)
    generator = ExLlamaGenerator(model, tokenizer, cache)

    return model, tokenizer, generator

def get_description_embedding(description, tokenizer, generator):
    inputs = tokenizer.encode(description)
    generator.gen_begin_reuse(inputs)
    embedding = generator.gen_single_token().detach().numpy().flatten()  # <--- Change I made
    return embedding

def preprocess_observation(observation, llm):
    user_id = np.array([observation['user_id']])
    user_gender = np.array([observation['user_gender']])
    user_age = np.array([observation['user_age']])  # Ensure this is 1-dimensional
    
    # Get the description embedding using the new method
    description_embedding = llm.tokenizer.encode(observation['user_description'])[0]
    description_embedding = np.array(description_embedding).flatten()  # <--- Change I made

    items_interact = observation['items_interact']
    if isinstance(items_interact, list):
        items_interact = np.array(items_interact).flatten()
    elif isinstance(items_interact, tuple):
        items_interact = np.array(items_interact).flatten()
    
    # Check dimensions before concatenation
    print(f"user_id shape: {user_id.shape}")
    print(f"user_gender shape: {user_gender.shape}")
    print(f"user_age shape: {user_age.shape}")
    print(f"description_embedding shape: {description_embedding.shape}")
    print(f"items_interact shape: {items_interact.shape}")

    flat_observation = np.concatenate([user_id, user_gender, user_age, description_embedding, items_interact])
    
    # Ensure the flat_observation has the required embedding_dim length
    if flat_observation.shape[0] < models[0]['embedding_dim']:
        padding = np.zeros(models[0]['embedding_dim'] - flat_observation.shape[0])  # <--- Change I made
        flat_observation = np.concatenate([flat_observation, padding])  # <--- Change I made
    else:
        flat_observation = flat_observation[:models[0]['embedding_dim']]


    
    return torch.tensor(flat_observation).float().unsqueeze(0)  # Convert to tensor and add batch dimension

def get_news_article_id(model, flat_observation):
    action, _states = model.predict(flat_observation, deterministic=True)
    return action

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    models = get_available_models()
    model_dir = os.path.join(base_path, models[0]["dir_name"])
    model_path = os.path.join(model_dir, 'model.zip')
    print("Model path:", model_path)
    
    # Load the LLM once and reuse it
    llm = load_LLM(models[0]['llm_model'])
    print(f"LLM {models[0]} Loaded")
    
    args = argparse.Namespace(**models[0])
    env = get_enviroment_from_args(llm, args)

    ob = env.reset()[0]  # Extract the first part of the observation tuple

    # Example observation structure
    # gender = 0 if self._user.gender == "M" else 1  in code in env.py
    observation = [
        {
            'user_id': 123,
            'user_gender': 1,
            'user_age': 45,
            'user_description': 'Interested in sports and finance news',
            'items_interact': ([3628, 9], [21023, 9])
        },
    ]

    flat_ob = preprocess_observation(observation[0], llm)
    
    # Load the trained A2C model
    model_path = os.path.join(base_path, models[0]['dir_name'], 'model.zip')
    model = A2C.load(model_path)

    print("Model loaded")

    # Get the recommended news article ID
    article_id = get_news_article_id(model, flat_ob)
    print("Recommended News Article ID:", article_id)
