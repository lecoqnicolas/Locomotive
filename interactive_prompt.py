import argparse
from typing import NoReturn
import time
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
                              device=config.device,
                              prompt_file=config.prompt,
                              batch_size=config.batch_size,
                              output_parser=config.response_parsing_method, prompt_ignore=config.ignore_prompt,
                              use_context=config.use_context, separateur_context=config.separateur_context,
                              context_window=config.context_window)


    print("Starting interactive mode")
    if params.manual_selection:
        # user-defined selection of source and destination languages
        config.src_name = input("Please select your source language :")
        config.src_code = config.src_name
        print(f"> {config.src_code} selected")
        config.tgt_name = input("Please select your target language :")
        config.tgt_code = config.tgt_name
        print(f"> {config.tgt_code} selected")

    if params.gui:
        # Gradio
        translator = get_gui(model, config.llm_model)
        translator.launch()
    else:
        # Command line
        while True:
            try:
                text = input(f"({config.src_code})> ")
            except KeyboardInterrupt:
                print("")
                exit(0)
            start_time = time.time()
            translated_text = model.transform([text], config.src_name, config.tgt_name)
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Response time: {response_time:.4f} seconds")
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
    parser.add_argument('--gui', action="store_true", help="Enable gradio-powered gui", default=False)
    parser.add_argument('--manual_selection', action='store_true', help="Manual lang selection")
    args = parser.parse_args()

    main(args)
