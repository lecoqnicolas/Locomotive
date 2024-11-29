"""
This file updates the env location for all pbtxt found under ROOT_DIRECTORY.

To optimize ram and mermoery consumption, all our models are using the same env.
This implies at least one of the model deployed must have an env (we can chose which one in deploy.py).
Then this script can be used to make all the models configuration link to the chosen env.
"""

import os
import re

ROOT_DIRECTORY = "./models"
NEW_PATH = "/models/madlad/traduction_env"


def replace_execution_env_path(root_dir, new_value):
    """
    Replaces the value of EXECUTION_ENV_PATH in all config.pbtxt files under a directory tree.

    Args:
        root_dir (str): The root directory to start the search.
        new_value (str): The new string value to replace after 'string_value'.

    Returns:
        None
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "config.pbtxt":
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()

                    # Replace the EXECUTION_ENV_PATH value
                    pattern = r'(key:\s*"EXECUTION_ENV_PATH".*?string_value:\s*")[^"]*'
                    replacement = r'\1' + new_value
                    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

                    # Write the updated content back to the file if changes were made
                    if updated_content != content:
                        with open(file_path, 'w') as file:
                            file.write(updated_content)
                        print(f"Updated: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    replace_execution_env_path(ROOT_DIRECTORY, NEW_PATH)
