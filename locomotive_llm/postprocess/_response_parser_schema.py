import logging
from enum import Enum, auto
from ._response_parsing_methods import _clean_output, _clean_output_json, _clean_translation_output
from typing import Callable


class LlmResponseParser(Enum):
    # User available schema
    keep_first_line = auto()
    bracket_regexp = auto()
    json = auto()


METHOD_MAPPING = {
    LlmResponseParser.keep_first_line: _clean_output,
    LlmResponseParser.bracket_regexp: _clean_translation_output,
    LlmResponseParser.json: _clean_output_json
}


def get_output_parsing_method(method: LlmResponseParser | str) -> Callable:
    if isinstance(method, str):
        method = LlmResponseParser[method]
    try:
        return METHOD_MAPPING[method]
    except KeyError:
        logging.error(f" {method} is not a valid llm output parsing method, possible choices ares : "
                      f"{list(LlmResponseParser.__members__.keys())}")
