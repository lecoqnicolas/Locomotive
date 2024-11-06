import os.path
import shutil
import argparse
import logging
from pathlib import Path
import subprocess
import fileinput
import sys


SOURCE_DEPENDANCIES = {
    "document_trad": ["./locomotive_llm"],
    "sentence_trad": ["./locomotive_llm"]
}

CUSTOM_ENV = {
    "sentence_trad": "models/sentence_trad/traduction_env.tar.gz"
}

TRITON_CONFIG_FILE = "config.pbtxt"


def fix_pydantic_for_langchain3_triton(env_directory: Path):
    """
    Modify lenient_enval pydantic call to support
    """
    file = Path(env_directory) /'lib/python3.10/site-packages/pydantic/_internal/_typing_extra.py'
    new_line = "    except (NameError, TypeError):"
    line_to_replace = "    except NameError:"
    success = False
    for i, line in enumerate(fileinput.input(file, inplace=1)):
        if line_to_replace in line:
            line = line.replace(line_to_replace,new_line)
            success = True
        # print is redirected to the file
        print(line, end='')
    return success


POST_ENV_ACTION = {
    "sentence_trad": fix_pydantic_for_langchain3_triton
}


def deploy_triton(arg):
    logging.getLogger().setLevel(logging.INFO)
    target_repository = Path(arg.triton_model_repository)
    if not os.path.isdir(target_repository):
        logging.info("Triton directory not found, creating it")
        os.makedirs(target_repository, exist_ok=True)
    src_repository = Path(args.local_model_repository)

    for elmt in src_repository.glob("*"):
        if elmt.is_dir():
            if arg.model is not None and elmt.name != arg.model:
                logging.info(f"Looking for {arg.model} : Skipping local model {arg.model}")
                continue
            logging.info("----------------------")
            logging.info(f"Found local model {elmt.name}")
            if arg.version is None:
                versions = [file for file in os.listdir(elmt) if file.isdigit()]
            elif (elmt / arg.version).is_dir():
                versions = [arg.version]
            else:
                logging.error(f"Version {arg.version} for model {elmt.name}")
                versions = list()

            if len(versions) == 0:
                logging.info(f"Skipping model {elmt.name} as no valid version was found")
                continue
            logging.info(f"Deploying versions {versions} of {elmt.name} to triton model repository {target_repository}")

            dest_model_dir = target_repository / elmt.name
            for version in versions:
                logging.info("------------")
                logging.info(f"Deploying version {version} to {target_repository}")
                dest_version_dir = dest_model_dir / version
                if dest_model_dir.is_dir():
                    shutil.rmtree(dest_version_dir)
                shutil.copytree(elmt/version, dest_version_dir)
                if elmt.name in SOURCE_DEPENDANCIES:
                    for dep in SOURCE_DEPENDANCIES[elmt.name]:
                        dep = Path(dep)
                        logging.info(f"Updating dependency {dep.name} for version {version}")
                        dest_dep_dir = dest_version_dir / dep.name
                        if dest_dep_dir.is_file():
                            shutil.rmtree(dest_dep_dir)
                        shutil.copytree(dep, dest_dep_dir)
                else:
                    logging.info(f"No source dependancy found for {elmt.name}")

            logging.info("----------------------")
            logging.info(f"Updating triton configuration file for model {elmt.name}")
            shutil.copy(elmt / TRITON_CONFIG_FILE,  dest_model_dir/ TRITON_CONFIG_FILE)
            if args.deploy_env and elmt.name in CUSTOM_ENV:
                logging.info(f"Deploying env {CUSTOM_ENV[elmt.name]} for model {elmt.name}")
                if (dest_model_dir / CUSTOM_ENV[elmt.name]).is_file():
                    shutil.rmtree(dest_model_dir / CUSTOM_ENV[elmt.name])

                shutil.copy(CUSTOM_ENV[elmt.name], dest_model_dir / Path(CUSTOM_ENV[elmt.name]).name)
                if args.extract_env:
                    prefix = Path(CUSTOM_ENV[elmt.name]).stem.split('.')[0]
                    logging.info(f"Extracting env {prefix} to {dest_model_dir/prefix}")
                    if (dest_model_dir / prefix).is_dir():
                        shutil.rmtree(dest_model_dir / prefix, ignore_errors=True)
                    os.makedirs(dest_model_dir/prefix, exist_ok=True)
                    subprocess.run(["tar", "-xvf", dest_model_dir / Path(CUSTOM_ENV[elmt.name]).name,
                                    "-C", dest_model_dir/prefix])
                    os.remove(dest_model_dir / Path(CUSTOM_ENV[elmt.name]).name)
                    if elmt.name in POST_ENV_ACTION:
                        result = POST_ENV_ACTION[elmt.name](dest_model_dir/prefix)
                        if result:
                            logging.info("Post env installation actions performed succesfully")
                        else:
                            logging.info("Post env installation actions failed")
            else:
                logging.info(f"No custom env specified for model {elmt.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate a document using TowerLLM model')

    parser.add_argument('--triton_model_repository',
                        type=str,
                        default="/models",
                        help='Path to triton model directory')

    parser.add_argument('--local_model_repository',
                        type=str,
                        default="./models",
                        help='Path to local models to deploy')

    parser.add_argument('--version',
                        type= int,
                        default= None,
                        help='If specified, only deploy a specific version of the models')

    parser.add_argument('--model',
                        type=str,
                        default=None,
                        help='If provided, only deploy a specifc model')

    parser.add_argument("--deploy_env",
                        type=bool,
                        default=False,
                        help= "if provided, will redeploy the conda env. Location is in the corresponding model directory")
    parser.add_argument("--extract_env", type=bool, default=True, help="If True, provided envs will be deployed "
                                                                       "extracted")
    args = parser.parse_args()

    deploy_triton(args)
