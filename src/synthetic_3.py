import pandas as pd
from synthcity.benchmark import Benchmarks
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


# Paths to get the original data and to save the generated data
path_in = r"C:\Users\Document\quindata_Gen\synthcity_env\data\input\willow_dataset_complete.xlsx"
path_out = "C:/Users/Document/quindata_Gen/synthcity_env/data/output/"
nb_rows = 100  # number of rows to load from the original data
nb_output_rows = 20 # number of rows to output from the synthetic data
synthetic_size = nb_output_rows

# Load real (Willow) dataset
def load_data(path_in, nb_rows):
    willow_data = pd.read_excel(path_in, nrows=nb_rows)
    usecols = list(set(willow_data.columns))  # Avoid redundant column selection
    return pd.read_excel(path_in, nrows=nb_rows, usecols=usecols)

# Separate features and target variable
def prepare_data(data):
    data = data.copy() # Avoid modifying original data
    data = data.dropna() # drop nulls
    X = data.drop("price", axis=1)  # More concise column drop
    y = data["price"]
    return X, y


def evaluate_and_generate(model_name, loader, synthetic_size, model_params=None):
    """
    Evaluates a model and generates synthetic data, handling different model parameters.

    Args:
        model_name (str): Name of the model to use (e.g: "ctgan", "tvae", "decaf").
        loader (GenericDataLoader): Data loader object containing the data.
        synthetic_size (int): Number of synthetic data points to generate.
        model_params (dict, optional): Dictionary of parameters specific to the model.
            Defaults to None.

    Returns:
        tuple: A tuple containing the evaluation score and the generated
                synthetic data as a pandas DataFrame.
    """

    # Perform evaluation
    score = Benchmarks.evaluate(
        [(f"GENERATOR_{model_name}", model_name, {})],
        loader,
        synthetic_size=synthetic_size,
        metrics={"sanity": ["data_mismatch"]},
    )

    # Get the model instance
    model = Plugins().get(model_name)

    # Handle model-specific parameters
    if model_params:
        model_fit_function = getattr(model, "fit_predict", None)  # Default function

        # ctgan parameters
        if model_name == "ctgan":
            model_fit_function = getattr(model, "fit", None)  # ctgan might use 'fit'
            model_params = {
                "n_iter": model_params.get("n_iter", 1),
                "generator_n_units_hidden": model_params.get("generator_n_units_hidden", 10),
                "discriminator_n_units_hidden": model_params.get("discriminator_n_units_hidden", 10),
            }

        # tvae parameters
        elif model_name == "tvae":
            model_params = {
                "n_iter": model_params.get("n_iter", 1),
                "n_units_embedding": model_params.get("n_units_embedding", 10),
                "decoder_n_units_hidden": model_params.get("decoder_n_units_hidden", 10),
                "encoder_n_units_hidden": model_params.get("encoder_n_units_hidden", 10),
            }

        # decaf parameters
        elif model_name == "decaf":
            model_params = {
                "n_iter": model_params.get("n_iter", 1),
                "n_iter_baseline": model_params.get("n_iter_baseline", 50),
                "generator_n_units_hidden": model_params.get("generator_n_units_hidden", 10),
                "discriminator_n_units_hidden": model_params.get("discriminator_n_units_hidden", 10),
                "struct_learning_n_iter": model_params.get("struct_learning_n_iter", 50),
            }

        # Use model-specific function with parameters (if available)
        if model_fit_function:
            model = model_fit_function(loader, synthetic_size, **model_params)
        else:
            # Fallback to generic fit_predict if model-specific function not found
            model = model.fit(loader, synthetic_size)
    else:
        # No model parameters provided, use default fit_predict
        model = model.fit(loader, synthetic_size)

    # Try converting to DataFrame
    try:
        data = model.generate(count=nb_output_rows).dataframe()
    except:
        print("CONVERSION TO DATAFRAME FAILED!")  # Keep the data as is if conversion fails
        exit()

    return score, data


if __name__ == "__main__":
    # call functions to get data
    willow_data = load_data(path_in, nb_rows)
    X, y = prepare_data(willow_data)
    loader = GenericDataLoader(X, target_column="price")
    models = ["ctgan", "tvae","decaf"]

    for model in models:
        print(f"\n\n TRAINING GENERATOR {model}...")
        # Evaluate each model
        score, syn_data = evaluate_and_generate(model, loader, synthetic_size)
        # save generated data
        print(f"\n\n BELOW THE sanity:data_mismatch SCORE OF {model} GENERATOR: \n")
        print(score)
        print(f"\n\n SAVING GENERATED DATA FROM THE MODEL {model}...")
        syn_data.to_csv(path_out + f'{model}_synth_dataset.csv', index=False)
        print("\n\n SUCCESSFUL SAVED!")

