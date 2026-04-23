import yaml
import numpy as np

def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_metric_tensors(cfg, seed: int = 0):
    """Returns random left/right metric tensors of shape (N, M, M).
    Placeholder until real LLR calculations are wired in.
    """
    np.random.seed(seed)
    nu = cfg["tbcc_config"]["V"]
    m  = cfg["bch_config"]["M"]
    M  = 2 ** (nu + m)   # total trellis states
    N  = cfg["bch_config"]["K"] + cfg["bch_config"]["M"]
    left  = np.random.rand(N, M, M).astype(np.float32)
    right = np.random.rand(N, M, M).astype(np.float32)
    return left, right, M, N
