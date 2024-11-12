import logging
from enum import Enum, auto
from ._response_parsing_methods import clean_output, clean_output_json, clean_translation_output, identity
from typing import Callable


class LlmResponseParser(Enum):
    # User available schema
    keep_first_line = auto()
    bracket_regexp = auto()
    json = auto()
    identity = auto()


METHOD_MAPPING = {
    LlmResponseParser.keep_first_line: clean_output,
    LlmResponseParser.bracket_regexp: clean_translation_output,
    LlmResponseParser.json: clean_output_json,
    LlmResponseParser.identity: identity
}


def get_output_parsing_method(method: LlmResponseParser | str) -> Callable:
    if isinstance(method, str):
        method = LlmResponseParser[method]
    try:
        return METHOD_MAPPING[method]
    except KeyError:
        logging.error(f" {method} is not a valid llm output parsing method, possible choices ares : "
                      f"{list(LlmResponseParser.__members__.keys())}")
