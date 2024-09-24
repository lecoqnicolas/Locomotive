import argparse

import torch

from locomotive_llm.load import load_config
from locomotive_llm.model import TowerLlmPipeline, TowerLlmPipelineLangChain, TowerInstructPipelineLangChain

torch.cuda.empty_cache()


def main(params: argparse.Namespace) -> None:
    # Load model configuration
    config = load_config(params.config, params.reverse)
    if params.use_towerinstruct and params.use_langchain:
        model = TowerInstructPipelineLangChain(device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu")
    elif params.use_langchain:
            model = TowerLlmPipelineLangChain(device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu")
    else:
        model = TowerLlmPipeline(device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu")
    
    print("Starting interactive mode")
    while True:
        try:
            text = input(f"({config['from']['code']})> ")
        except KeyboardInterrupt:
            print("")
            exit(0)

        translated_text = model.transform([text], config['from']["name"], config["to"]["name"])
        print(f"({config['to']['code']})> {translated_text[0]}")


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
    parser.add_argument('--use_langchain',
                        action='store_true',
                        help='Use LangChain for the translation pipeline. Default: %(default)s')
    parser.add_argument('--use_towerinstruct',
                        action='store_true',
                        help='Use LangChain for the translation pipeline. Default: %(default)s')
    
    args = parser.parse_args()

    main(args)
