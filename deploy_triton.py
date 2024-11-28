import os.path
import shutil
import argparse
import logging
from pathlib import Path
import subprocess
import fileinput


SOURCE_DEPENDANCIES = {
    "document_trad": ["./locomotive_llm"],
    "sentence_trad": ["./locomotive_llm"],
    "madlad": ["./locomotive_llm","artifacts/madlad400-10b-mt"],
    "en_fr_seq2seq": ["./seq2seq_model/inference.py", "./seq2seq_model/translate-en_fr-1_10/"],
    "ar_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-ar_fr-1_2/'],
    "de_en_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-de_en-1_4/'],
    "de_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-de_fr-1_2/'],
    "en_de_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-en_de-1_4/'],
    "en_hy_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-en_hy-1_2/'],
    "es_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-es_fr-1_2/'],
    "fr_de_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_de-1_2/'],
    "fr_en_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_en-1_10/'],
    "fr_es_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_es-1_2/'],
    "fr_pt_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_pt-1_2/'],
    "fr_ru_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_ru-1_2/'],
    "fr_zh_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-fr_zh-1_2/'],
    "hy_en_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-hy_en-1_2/'],
    "pt_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-pt_fr-1_2/'],
    "ru_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-ru_fr-1_2/'],
    "tr_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-tr_fr-1_2/'],
    "zh_fr_seq2seq": ['./seq2seq_model/inference.py', './seq2seq_model/translate-zh_fr-1_2/'],
    "onnx_translation_model": ["./tower_onnx_2"],
    "tower_module":["./artifacts/TowerInstruct-Mistral-7B-v0.2"],
    
}

CUSTOM_ENV = {
    "sentence_trad": "models/sentence_trad/traduction_env.tar.gz",
    "sentence_trad_prepro": "models/sentence_trad_prepro/traduction_env.tar.gz"
}

TRITON_CONFIG_FILE = "config.pbtxt"
TRITON_CERTIFICATES_FOLDER = Path("/opt/tritonserver/certs")
CERTIFICATES_FILES = ["server_localhost.crt", "server_localhost.key", "ca_localhost.crt"]


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
    "sentence_trad": fix_pydantic_for_langchain3_triton,
    "sentence_trad_prepro": fix_pydantic_for_langchain3_triton,
    "sentence_trad_prepro_docs": fix_pydantic_for_langchain3_triton
}


def deploy_triton(arg):
    logging.getLogger().setLevel(logging.INFO)
    target_repository = Path(arg.triton_model_repository)
    if not os.path.isdir(target_repository):
        logging.info("Triton directory not found, creating it")
        os.makedirs(target_repository, exist_ok=True)
    src_repository = Path(args.local_model_repository)

    logging.info("Updating certificates")
    if not os.path.isdir(TRITON_CERTIFICATES_FOLDER):
        logging.info("Certificate folder not found, creating it")
        os.makedirs(TRITON_CERTIFICATES_FOLDER, exist_ok=True)
    for cert_file in CERTIFICATES_FILES:
        if not os.path.isfile(arg.certificates_dir / cert_file):
            logging.warning(f"Certificate file {arg.certificates_dir / cert_file} not found, skipped")
        else:
            shutil.copyfile(arg.certificates_dir / cert_file, TRITON_CERTIFICATES_FOLDER / cert_file)

    for elmt in src_repository.glob("*"):
        if elmt.is_dir():
            if arg.model is not None and elmt.name != arg.model:
                logging.info(f"Looking for {arg.model} : Skipping local model {elmt.name}")
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
                logging.info(f"Skipping model {elmt.name} : no valid version was found")
                continue
            logging.info(f"Deploying versions {versions} of {elmt.name} to triton model repository {target_repository}")

            dest_model_dir = target_repository / elmt.name
            for version in versions:
                logging.info("------------")
                logging.info(f"Deploying version {version} to {target_repository}")
                dest_version_dir = dest_model_dir / version
                if dest_version_dir.is_dir():
                    shutil.rmtree(dest_version_dir)
                #os.makedirs(dest_version_dir, exist_ok=True)
                shutil.copytree(elmt/version, dest_version_dir)
                if elmt.name in SOURCE_DEPENDANCIES:
                    for dep in SOURCE_DEPENDANCIES[elmt.name]:
                        dep = Path(dep)
                        logging.info(f"Updating dependency {dep.name} for version {version}")
                        dest_dep_dir = dest_version_dir / dep.name
                        if dest_dep_dir.is_file():
                            shutil.rmtree(dest_dep_dir)
                        if dep.is_file():
                            shutil.copyfile(dep, dest_dep_dir)
                        else:
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

    parser.add_argument('--certificates_dir',
                        type=Path,
                        default='./certs',
                        help="Servers certificates")

    parser.add_argument("--deploy_env",
                        type=bool,
                        default=False,
                        help= "if provided, will redeploy the conda env. Location is in the corresponding model directory")
    parser.add_argument("--extract_env", type=bool, default=True, help="If True, provided envs will be deployed "
                                                                       "extracted")
    args = parser.parse_args()

    deploy_triton(args)
