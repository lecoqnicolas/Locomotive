import json
import re


def _clean_output(output, prompt):
    cleaned_output = output.replace(prompt, "").strip()
    if "\n" in cleaned_output:
        cleaned_output = cleaned_output.split("\n")[0].strip()

    return cleaned_output


def _clean_translation_output(output, prompt):
    # Find translations in the output
    matches = re.findall(r'"translated_text":\s*"([^"]*)"', output)

    # Extract translations
    translations = [match for match in matches]

    # Create a valid JSON response
    if translations:
        # Take the last translation and remove any quotes
        cleaned_translation = translations[1].replace('"', '').strip()  # Remove quotes
        return cleaned_translation  # Return cleaned translation

    return None


def _clean_output_json(output, prompt):
    json_output = ""

    # Find the first and last braces to extract potential JSON
    output = output.replace("'", '"')
    matches = re.findall(r'\{.*?\}', output, re.DOTALL)
    if matches:
        json_output = matches[-1].strip()  # Take the last matched JSON structure

    if json_output:
        try:
            parsed_output = json.loads(json_output)

            return parsed_output.get("translated_text", "").strip().replace(prompt, "")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON output: {e}, output was: {json_output}")
            return ""

    print(f"No valid JSON found in output: {output}")
    return ""

