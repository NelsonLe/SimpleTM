import argparse
from custom.model.CustomTM import CustomTM

def train():
    pass

def test():
    pass

def main():
    parser = argparse.ArgumentParser(description="CustomTM")

    # CustomTM model arguments
    # @ Nelson, can variables just be inferred from the dataset?
    # not sure how preprocessing works - TGR
    parser.add_argument(
        "--variables", type=int, required=True,
        help="Number of variables in multivariate time series data"
    )
    parser.add_argument(
        "--length", type=int, default=96,
        help="Lookback window length"
    )
    parser.add_argument(
        "--pseudo_length", type=int, default=32,
        help="Dimensionality of pseudo-tokens"
    )
    parser.add_argument(
        "--prediction_length", type=int, default=96,
        help="Forecasting window length"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout ratio for non-attention feedforward layers"
    )
    parser.add_argument(
        "--m", type=int, default=2,
        help="Number of decomposition levels for Wavelet transform"
    )
    parser.add_argument(
        "--learnable_wavelets", action="store_true", dest="learnable_wavelets",
        help="Allow wavelet convolutions to be learnable"
    )
    parser.add_argument(
        "--wv", type=str, default="db1",
        help="Wavelet function used for decomposition/initialization"
    )
    parser.add_argument(
        "--pad_mode", type=str, default="circular",
        choices=["constant", "reflect", "replicate", "circular"],
        help="Padding mode for constant-length wavelet decomposition"
    )
    parser.add_argument(
        "--inverted", action="store_true", dest="inverted",
        help="Apply linear projection after wavelet decomposition in original time domain"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Wedge product weighting in geometric self-attention"
    )
    parser.add_argument(
        "--scale", type=float, default=None, 
        help="Scaling factor for query-key product in self-attention"
    )
    parser.add_argument(
        "--attention_dropout", type=float, default=0.1,
        help="Dropout ratio for attention layers"
    )
    parser.add_argument(
        "--normalize", action="store_true", dest="normalize",
        help="Feed normalized data into the model, then unnormalize outputs"
    )
    parser.add_argument(
        "--transformer_layers", type=int, default=1,
        help="Number of SWT/Attention/ISWT/Feedforward blocks"
    )
    parser.add_argument(
        "--is_geometric", action="store_true", dest="is_geometric",
        help="Use geometric (as opposed to vanilla) attention"
    )
    parser.add_argument(
        "encoder_activation", type=str, default="gelu",
        choices=["relu", "gelu"],
        help="Activation function to use in feedforward layers"
    )
    parser.add_argument(
        "--feedforward_dim", type=int, default=32,
        help="Hidden dimension of feedforward layers"
    )

    #TODO: add the following flags...
    # l1 weight
    # early stopping rounds/epochs
    # learning rate
    # seeding & iteration numbers
    # dataset
    # among various others...
    # not sure exactly what is/isn't extraneous
    # - TGR
    configs = parser.parse_args

    # create model
    model = CustomTM(configs)

if __name__ == "__main__":
    main()