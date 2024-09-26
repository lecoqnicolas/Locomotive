import itertools
import json
import logging
import os

import numpy as np
import torch
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file

from comet import download_model, load_from_checkpoint
from comet.models.utils import split_sequence_into_sublists
from dataclasses import dataclass
from pathlib import Path

torch.set_float32_matmul_precision('high')

@dataclass
class CometConfig:
    sources: list[Path]
    translations: list[Path]
    references: Path
    sacrebleu_dataset: str
    batch_size: int = 16
    gpus: int = 1
    quiet: bool = False
    only_system: bool = False
    to_json: str = ""
    model: str = "Unbabel/wmt22-comet-da"
    model_storage_path: str = None
    num_workers: int = None
    disable_cache: bool = False
    disable_length_batching: bool = False
    print_cache_info: bool = False


def score_command(cfg: CometConfig):

    if cfg.quiet:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.ERROR)

    seed_everything(1)
    if cfg.sources is None and cfg.sacrebleu_dataset is None:
        logging.error(f"You must specify a source (-s) or a sacrebleu dataset (-d)")

    if cfg.sacrebleu_dataset is not None:
        if cfg.references is not None or cfg.sources is not None:
            logging.error(
                f"Cannot use sacrebleu datasets (-d) with manually-specified datasets (-s and -r)"
            )

        try:
            testset, langpair = cfg.sacrebleu_dataset.rsplit(":", maxsplit=1)
            cfg.sources = Path_fr(get_source_file(testset, langpair))
            cfg.references = Path_fr(get_reference_files(testset, langpair)[0])
        except ValueError:
            logging.error(
                "SacreBLEU testset format must be TESTSET:LANGPAIR, e.g., wmt20:de-en"
            )
        except Exception as e:
            import sys

            print("SacreBLEU error:", e, file=sys.stderr)
            sys.exit(1)

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    else:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)

    model = load_from_checkpoint(model_path)
    model.eval()

    if model.requires_references() and (cfg.references is None):
        logging.error(
            "{} requires -r/--references or -d/--sacrebleu_dataset.".format(cfg.model)
        )

    if not cfg.disable_cache:
        model.set_embedding_cache()

    with open(cfg.sources, encoding="utf-8") as fp:
        sources = [line.strip() for line in fp.readlines()]

    translations = []
    for path_fr in cfg.translations:
        with open(path_fr, encoding="utf-8") as fp:
            translations.append([line.strip() for line in fp.readlines()])

    if cfg.references is not None:
        with open(cfg.references, encoding="utf-8") as fp:
            references = [line.strip() for line in fp.readlines()]
        data = {
            "src": [sources for _ in translations],
            "mt": translations,
            "ref": [references for _ in translations],
        }
    else:
        data = {"src": [sources for _ in translations], "mt": translations}

    if cfg.gpus > 1:
        # Flatten all data to score across multiple GPUs
        for k, v in data.items():
            data[k] = list(itertools.chain(*v))

        data = [dict(zip(data, t)) for t in zip(*data.values())]
        outputs = model.predict(
            samples=data,
            batch_size=cfg.batch_size,
            gpus=cfg.gpus,
            progress_bar=(not cfg.quiet),
            accelerator="auto",
            num_workers=cfg.num_workers,
            length_batching=(not cfg.disable_length_batching),
        )
        seg_scores = outputs.scores
        if "metadata" in outputs and "error_spans" in outputs.metadata:
            errors = outputs.metadata.error_spans
        else:
            errors = []

        if len(cfg.translations) > 1:
            seg_scores = np.array_split(seg_scores, len(cfg.translations))
            sys_scores = [sum(split) / len(split) for split in seg_scores]
            data = np.array_split(data, len(cfg.translations))
            errors = split_sequence_into_sublists(errors, len(cfg.translations))
        else:
            sys_scores = [
                outputs.system_score,
            ]
            seg_scores = [
                seg_scores,
            ]
            errors = [errors, ]
            data = [
                np.array(data),
            ]
    else:
        # If not using Multiple GPUs we will score each system independently
        # to maximize cache hits!
        seg_scores, sys_scores, errors = [], [], []
        new_data = []
        for i in range(len(cfg.translations)):
            sys_data = {k: v[i] for k, v in data.items()}
            sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
            new_data.append(np.array(sys_data))
            outputs = model.predict(
                samples=sys_data,
                batch_size=cfg.batch_size,
                gpus=cfg.gpus,
                progress_bar=(not cfg.quiet),
                accelerator="cpu" if cfg.gpus == 0 else "auto",
                num_workers=cfg.num_workers,
                length_batching=(not cfg.disable_length_batching),
            )
            seg_scores.append(outputs.scores)
            sys_scores.append(outputs.system_score)
            if "metadata" in outputs and "error_spans" in outputs.metadata:
                errors.append(outputs.metadata.error_spans)
        data = new_data

    files = [path_fr.rel_path for path_fr in cfg.translations]
    data = {file: system_data.tolist() for file, system_data in zip(files, data)}
    for i in range(len(data[files[0]])):  # loop over (src, ref)
        for j in range(len(files)):  # loop of system
            data[files[j]][i]["COMET"] = seg_scores[j][i]
            if errors and errors[j] and errors[j][i]:
                data[files[j]][i]["errors"] = errors[j][i]

            if not cfg.only_system:
                print(
                    "{}\tSegment {}\tscore: {:.4f}".format(
                        files[j], i, seg_scores[j][i]
                    )
                )

    for j in range(len(files)):
        print("{}\tscore: {:.4f}".format(files[j], sys_scores[j]))

    if cfg.to_json != "":
        with open(cfg.to_json, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())
    return sys_scores, seg_scores