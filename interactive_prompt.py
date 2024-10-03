import argparse
from typing import NoReturn

import torch

from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline
from locomotive_llm.ui import get_gui

torch.cuda.empty_cache()


def main(params: argparse.Namespace) -> NoReturn:
    # Load model configuration
    config = load_config(params.config, params.reverse)
    pipeline_class = get_pipeline(config)
    model = pipeline_class(model_id=config.llm_model,
                           device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                           prompt_file=config.prompt,
                           max_tokens=config.max_token)

    print("Starting interactive mode")
    if params.manual_selection:
        # user selection of langs
        config.src_name = input("Please select your source langage :")
        config.src_code = config.src_name
        print(f"> {config.src_code} selected")
        config.tgt_name = input("Please select your target langage :")
        config.tgt_code = config.tgt_name
        print(f"> {config.tgt_code} selected")

    if params.gui:
        translator = get_gui(model, config.llm_model)
        translator.launch()

    else:
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
    parser.add_argument('--gui', action="store_true", help="Enable gradio-powered gui", default=True)
    parser.add_argument('--manual_selection', action='store_true', help="Manual lang selection")
    args = parser.parse_args()

    main(args)
