import optuna
import subprocess


from algorithms.logging_config  import get_logger

logger = get_logger("suber_logger")


def objective(trial):
    # Define the parameter search space with appropriate models
##    llm_model = trial.suggest_categorical('llm_model', [
##        'TheBloke/Llama-2-7b-Chat-GPTQ', 'TheBloke/vicuna-7B-v1.3-GPTQ',
##        'TheBloke/vicuna-13b-v1.3.0-GPTQ', 'TheBloke/vicuna-7B-v1.5-GPTQ',
##        'TheBloke/vicuna-13B-v1.5-GPTQ', 'gpt-3.5-turbo-0613',
##        'gpt-4o', 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ',
##    ])
    llm_model = trial.suggest_categorical('llm_model', [
        'TheBloke/vicuna-13B-v1.5-GPTQ',
        'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ',
    ])


    llm_rater = trial.suggest_categorical('llm_rater', [
        '2Shot_system_our', '1Shot_system_our', '0Shot_system_our',
        '0Shot_system_our_1_10', '1Shot_system_our_1_10', '2Shot_system_our_1_10',
        '2Shot_invert_system_our', '1Shot_invert_system_our', 
    ])
    items_retrieval = trial.suggest_categorical('items_retrieval', [
        'last_3', 'most_similar_3_title','most_similar_3_abstract', 'none', 'simple_3'
    ])
    user_dataset = trial.suggest_categorical('user_dataset', ['mind'])
    perturbator = trial.suggest_categorical('perturbator', ['none', 'gaussian', 'greedy'])
    reward_shaping = trial.suggest_categorical('reward_shaping', ['identity', 'exp_decay_time', 'random_watch'])
    #seed = trial.suggest_int('seed', 0, 100)
    seed = 42
    #model_device = trial.suggest_categorical('model_device', ['cpu', 'cuda'])
    model_device = "cuda:0"
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.99])
    embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])

    #learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3])  # Use a list of predefined learning rates
 
    # Creating a custom study name based on parameters
    #study_name = f"llm-{llm_model}-rater-{llm_rater}-device-{model_device}-seed-{seed}"
    print(f"llm_model: {llm_model}, llm_rater: {llm_rater}, items_retrieval: {items_retrieval}, user_dataset: {user_dataset}, perturbator: {perturbator}, reward_shaping: {reward_shaping}, seed: {seed}, model_device: {model_device}, gamma: {gamma}, embedding_dim: {embedding_dim}, learning_rate: {learning_rate}")
    logger.info(f"llm_model: {llm_model}, llm_rater: {llm_rater}, items_retrieval: {items_retrieval}, user_dataset: {user_dataset}, perturbator: {perturbator}, reward_shaping: {reward_shaping}, seed: {seed}, model_device: {model_device}, gamma: {gamma}, embedding_dim: {embedding_dim}, learning_rate: {learning_rate}")
        # Build the command with the selected parameters
    command = [
        'python3', '-m', 'algorithms.mind.CF_train_A2C',
        '--llm-model', llm_model,
        '--llm-rater', llm_rater,
        '--items-retrieval', items_retrieval,
        '--user-dataset', user_dataset,
        '--perturbator', perturbator,
        '--reward-shaping', reward_shaping,
        '--seed', str(seed),
        '--model-device', model_device,
        '--gamma', str(gamma),
        '--embedding-dim', str(embedding_dim),
        '--learning_rate', str(learning_rate),
    ]
    
    # Run the command and capture the output
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info("Result: {}".format(result))
        output = result.stdout
        logger.info(output)
    except subprocess.CalledProcessError as e:
        logger.error("Command '{}' returned non-zero exit status {}.".format(e.cmd, e.returncode))
        logger.error(e.output)
    except Exception as e:
        logger.error("An error occurred: {}".format(str(e)))
    #mean_reward = train_and_evaluate(llm_model, llm_rater, items_retrieval, user_dataset, perturbator, reward_shaping, seed, model_device, gamma, embedding_dim, learning_rate)

    #return mean_reward

    # Extract the relevant metric from the result (assuming the last line contains "Mean Reward: X.XX")
    metric = 0
    logger.info("Trying to get metrics")
    try:
        for line in output.split("\n"):
            if "Mean Reward" in line:
                metric = float(line.split()[-1])
                logger.info("Got the metrics {}".format(metric))
                break
    except Exception as e:
        print(f"Error extracting metric: {e}")
        logger.info(f"Error extracting metric: {e}")
        metric = 0  # Assign a default poor performance score in case of an error

    return metric

# Create a study and optimize the objective function
study = optuna.create_study(storage="sqlite:///db.sqlite3",study_name="MIND_SUBER",direction='maximize',load_if_exists = True)
study.optimize(objective, n_trials=50)

print(study.best_trial)
logger.info(study.best_trial)