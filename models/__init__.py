from importlib import import_module

__all__ = ["CoordDeepONet", "CoordMLP", "GPLFR", "KNN", "PPCAICM", "PCAMLP", "PCARidge", "TrainMean"]

_EXPORTS = {
    "CoordDeepONet": (".coord_deeponet", "CoordDeepONet"),
    "CoordMLP": (".coord_mlp", "CoordMLP"),
    "GPLFR": (".gplfr", "GPLFR"),
    "KNN": (".knn", "KNN"),
    "PPCAICM": (".ppca_icm", "PPCAICM"),
    "PCAMLP": (".pca_mlp", "PCAMLP"),
    "PCARidge": (".pca_ridge", "PCARidge"),
    "TrainMean": (".train_mean", "TrainMean"),
}


def __getattr__(name: str):
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)
