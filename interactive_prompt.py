import argparse

import torch

from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline
torch.cuda.empty_cache()


def main(params: argparse.Namespace) -> None:
    # Load model configuration
    config = load_config(params.config, params.reverse)
    pipeline_class = get_pipeline(config)
    model = pipeline_class(model_id=config.llm_model,
                           device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                           prompt_file=config.prompt,
                           batch_size=config.batch_size)

    print("Starting interactive mode")
    while True:
        try:
            text = input(f"({config.src_code})> ")
        except KeyboardInterrupt:
            print("")
            exit(0)

        translated_text = model.transform([text], config.src_name, config.tgt_name)
        print(f"({config.tgt_code})> {translated_text[0]}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Evaluate TowerLLM model')
    parser.add_argument('--config',
                        type=str,
                        default="config/config_en_fr.yml",
                        help='Path to model-config.json. Default: %(default)s')
    parser.add_argument('--reverse',
                        action='store_true',
                        help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
    parser.add_argument('--cpu',
                        action="store_true",
                        help='Force CPU use. Default: %(default)s')
    
    args = parser.parse_args()

    main(args)
