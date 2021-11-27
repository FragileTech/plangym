import warnings


warnings.filterwarnings(
    "ignore",
    message=" WARN: Box bound precision lowered by casting to float32",
)
warnings.filterwarnings(
    "ignore",
    message=" WARN: Box bound precision lowered by casting to float64",
)
