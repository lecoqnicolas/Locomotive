import logging

import torch
from comet import download_model, load_from_checkpoint


def eval_text_comet(translated_text, reference_text, src_text):
    """
    Evaluate a whole text using comet.
    """
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    data = [
        {"src": src_text, "mt": translated_text, "ref": reference_text}
    ]

    comet_score = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    logging.info(f"COMET score: {comet_score}")
    return comet_score
