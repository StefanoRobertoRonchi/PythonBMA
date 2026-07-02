from BMA_SSVS import Bayesian_MA_SSVS, PiP, TopModels, data_preparation
from scipy.stats import norm
import yaml 

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        conf = yaml.safe_load(f)

    target = conf["target"]
    features = conf["features"]


    try:
        y,X = data_preparation(filepath = conf["filepath"], data = conf["data"], features = features, target = target )
    except:
        raise KeyError("The dataset doesn't contained the features specified")

    # Model development
    models = Bayesian_MA_SSVS(y=y, X=X, 
                    n=conf["params"]["n"], burn_in= conf["params"]["burn_in"],
                    a=conf["params"]["a"], b=conf["params"]["b"], betas_sd_ratio = conf["params"]["betas_sd_ratio"],
                    c=conf["params"]["c"], p= conf["params"]["p"],
                    identity = conf["params"]["identity"], add_intercept = conf["params"]["add_intercept"],
                    normalize_x = conf["params"]["normalize_x"], 
                    normalize_y = conf["params"]["normalize_y"])

    # Print the Posterior Inclusion Probability
    print(PiP(models,X))

    # Most common 5 models
    print(TopModels(models,y,X))
    


