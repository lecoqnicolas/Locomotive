from ._response_parser_schema import get_output_parsing_method, LlmResponseParser


class BasicPostProcessor:
    def __init__(self, output_parsing_method: str| LlmResponseParser) -> None:
        self._parsing_method = get_output_parsing_method(output_parsing_method)
        self._output_answer_field = "generated_text"

    def transform(self, valid_mask: list[bool], input_prompts: list[str], outputs: list) -> list[str]:
        """
        Clean and add empty sentences for unvalid prompt, to preserve original order
        """
        results = []
        output_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                if output_idx < len(outputs) and self._output_answer_field in outputs[output_idx][0]:
                    cleaned_output = self._parsing_method(outputs[output_idx][0][self._output_answer_field],
                                                          input_prompts[output_idx])
                else:
                    cleaned_output = ""
                output_idx += 1
            else:
                cleaned_output = ""
            results.append(cleaned_output)
        return results