import argparse

import torch

from locomotive_llm.load import load_config
from locomotive_llm.model import TowerLlmPipeline

torch.cuda.empty_cache()


def main(params: argparse.Namespace) -> None:
    # Load model configuration
    config = load_config(params.config, params.reverse)
    model = TowerLlmPipeline(device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Starting interactive mode")
    while True:
        try:
            text = input(f"({config['from']['code']})> ")
        except KeyboardInterrupt:
            print("")
            exit(0)

        translated_text = model.transform(text, config['from']["name"], config["to"]["name"])
        print(f"({config['to']['code']})> {translated_text}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Evaluate TowerLLM model')
    parser.add_argument('--config',
                        type=str,
                        default="config_llm.json",
                        help='Path to model-config.json. Default: %(default)s')
    parser.add_argument('--reverse',
                        action='store_true',
                        help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
    parser.add_argument('--cpu',
                        action="store_true",
                        help='Force CPU use. Default: %(default)s')
    args = parser.parse_args()

    main(args)
