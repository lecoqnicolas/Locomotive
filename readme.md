## Installation
- Tested with cuda 12.4.1 on debian 12.
- Conda installation :
 
    conda env create -f conda.yaml
    conda activate traduction_env

## Deployement

Clone or get the latest version of the project locomotive_env.

The current deployement script expects to find our artifacts in ./artifacts.
You can either transfer the artifacts, or if you have an internet acces, download them from the huggingface
repository :

For direct download, inside ./articats, run the following commands (you need git and git lfs):

  
  git clone https://huggingface.co/google/madlad400-10b-mt
  git clone https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2


Alternatively, just transfer the artifacts from the machine we prodided :

  
  - scp -r ./artifacts USER@IP:~/locomotive/artifacts
    

For the artifacts provided by Nicolas, we kept the provided structure : seq2seqmodel must contain
one directory with Nicholas structure per model you want to deploy (ex : seq2seq_model/translate-ar_fr-1_2, ...):

      - scp -r ./seq2seq_model USER@IP:~/locomotive/seq2seq_model 
  

      
## RUNNING the server

  1) generate certificates (or provide your own in ./certs)

   sh ./scripts/gen_certs.sh

  2) deploy one or more models

  python deploy_triton.py --model en_fr_seq2seq

  3) run the server

  sh ./scripts/run_triton.sh

Logs are availables at /opt/tritonserver/logs.txt. To modify the place where the logs are stored, you can change
the "log_dir" argument in run_triton.sh


### Installation triton :
- For debian 12 (no docker install) :

    sudo apt update
    sudo apt-get install linux-headers-`uname -r`
    sudo update-grub
    sudo apt-get install git
    # installation de cuda 12.4.1
    sudo apt-get install gcc
    sudo apt-get install make
    sudo apt-get install software-properties-common
    #wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    #sudo sh cuda_12.4.1_550.54.15_linux.run --silent
    sudo add-apt-repository contrib
    sudo apt-get update
    sudo bash -c 'echo "deb http://ftp.de.debian.org/debian/ bookworm main contrib non-free non-free-firmware" >> /etc/apt/sources.list'
    sudo bash -c 'echo "deb-src http://ftp.de.debian.org/debian/ bookworm main contrib non-free non-free-firmware" >> /etc/apt/sources.list'
    sudo apt-get update
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-debian12-12-4-local_12.4.1-550.54.15-1_amd64.deb
    sudo dpkg -i cuda-repo-debian12-12-4-local_12.4.1-550.54.15-1_amd64.deb
    sudo cp /var/cuda-repo-debian12-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo add-apt-repository contrib
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    sudo apt-get install -y cuda-drivers
    wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-debian12-9.1.0_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-debian12-9.1.0_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-debian12-9.1.0/cudnn-local-0C975CD8-keyring.gpg /usr/share/keyrings/
    sudo add-apt-repository contrib
    sudo apt-get update
    sudo apt-get -y install cudnn
    # check nvidia-smi work
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
    tar -xzvf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
    export LD_LIBRARY_PATH=/home/debian/TensorRT-10.0.1.6/lib:$LD_LIBRARY_PATH
    # permanently
    sudo bash -c 'echo "export LD_LIBRARY_PATH=/home/debian/TensorRT-10.0.1.6/lib:\$LD_LIBRARY_PATH" >> /home/debian/.bashrc'
    # install python 3.10
    wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
    tar zxf Python-3.10.0.tgz
    cd Python-3.10.0/
    sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
    ./configure --prefix=/usr/local
    make
    sudo make install
    # install tensorrt
    cd ../TensorRT-10.0.1.6/python
    python3.10 -m pip install tensorrt-*-cp310-none-linux_x86_64.whl
    cd ../onnx_graphsurgeon
    python3.10 -m pip install onnx_graphsurgeon-0.5.0-py2.py3-none-any.whl
    # verify it works with compiling sample_onxx then running in ~/TensorRT-10.0.1.6/ : ./bin/sample_onnx_mnist
    #
    cd
    git clone https://github.com/triton-inference-server/server.git
    cd server
    # to work with cuda 12.4.1 (https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2470/release-notes/rel-24-05.html)
    git checkout r24.05
    python3.10 -m pip install requests
    sudo apt-get install -y --no-install-recommends \
        ca-certificates \
        autoconf \
        automake \
        build-essential \
        git \
        gperf \
        libre2-dev \
        libssl-dev \
        libtool \
        libcurl4-openssl-dev \
        libb64-dev \
        libgoogle-perftools-dev \
        patchelf \
        rapidjson-dev \
        scons \
        software-properties-common \
        pkg-config \
        unzip \
        wget \
        zlib1g-dev \
        libarchive-dev \
        libxml2-dev \
        libnuma-dev
    sudo rm -rf /var/lib/apt/lists/*
    python3.10 -m pip install --upgrade pip
    python3.10 -m pip install --upgrade wheel setuptools virtualenv
    # skipped docker install with python
    wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz
    cd /tmp && tar xzf boost.tar.gz
    sudo mv /tmp/boost_1_80_0/boost /usr/include/boost
    cd
    wget https://cmake.org/files/v3.27/cmake-3.27.7.tar.gz
    tar -xzvf cmake-3.27.7.tar.gz
    cd cmake-3.27.7/
    ./bootstrap
    make -j 20 # 20 = proc number
    sudo make install
    # cmake --version check version
    # server build
    cd ../server/
    # TEST libz2, libsql3, tk et libxml dev
    sudo apt install libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    # the two exports may not be necessary
    export TRITON_SERVER_VERSION=24.05
    export NVIDIA_TRITON_SERVER_VERSION=2.46.0
    #python3.10 build.py -v --no-container-build --build-dir=`pwd`/build --enable-all --dryrun
    sudo mkdir /opt/tritonserver
    sudo chown debian: /opt/tritonserver
    # install dcgm
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y datacenter-gpu-manager
    # TODO : add to bashrc
    export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-12/bin:$PATH"
    export CUDA_HOME=/usr/local/cuda-12
    # datacenter-gpu-manager=1:3.2.6
    mkdir -p /home/debian/server/build/tritonserver/build
    cd /home/debian/server/build/tritonserver/build
    cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/home/debian/server/build/tritonserver/install" "-DTRITON_VERSION:STRING=2.47.0" "-DTRITON_REPO_ORGANIZATION:STRING=https://github.com/triton-inference-server" "-DTRITON_COMMON_REPO_TAG:STRING=r24.06" "-DTRITON_CORE_REPO_TAG:STRING=r24.06" "-DTRITON_BACKEND_REPO_TAG:STRING=r24.06" "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r24.06" "-DTRITON_ENABLE_LOGGING:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" "-DTRITON_ENABLE_METRICS_GPU:BOOL=ON" "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=ON" "-DTRITON_ENABLE_GPU:BOOL=ON" "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=OFF" "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" "-DCMAKE_POLICY_DEFAULT_CMP0148=OLD" "-DCMAKE_POLICY_DEFAULT_CMP0115=OLD" "-DTRITON_ENABLE_TENSORRT:BOOL=ON" "-DPYBIND11_FINDPYTHON=ON" "-DPython_EXECUTABLE=/usr/local/bin/python3.10" /home/debian/server
    # add to cmake_install.cmake
        set(CMAKE_POLICY_DEFAULT_CMP0115 OLD)
        set(CMAKE_POLICY_DEFAULT_CMP0148 OLD)
    # IMPORTANT : MODIFY THE FILE
    # to find the file to modify if not at the same place : grep -Rnw . -e 'https://github.com/libevent/libevent.git'
    # emacs _deps/repo-third-party-src/CMakeLists.txt
    # replace   GIT_REPOSITORY "https://github.com/libevent/libevent.git"
    # GIT_TAG ...
    # with :
    # GIT_REPOSITORY "https://salsa.debian.org/debian/libevent.git"
    # GIT_TAG "upstream/2.1.12-stable"
    # END IMPORTANT
    cmake --build . --config Release -j60  -t install -- DCMAKE_POLICY_DEFAULT_CMP0115=OLD DCMAKE_POLICY_DEFAULT_CMP0148=OLD
    mkdir -p /opt/tritonserver/bin
    mkdir -p /opt/tritonserver/lib
    mkdir -p /opt/tritonserver/python
    mkdir -p /opt/tritonserver/include/triton
    sudo chown -R debian: /opt/tritonserver
    cp /home/debian/server/build/tritonserver/install/bin/tritonserver /opt/tritonserver/bin
    cp /home/debian/server/build/tritonserver/install/lib/libtritonserver.so /opt/tritonserver/lib
    cp /home/debian/server/build/tritonserver/install/python/tritonserver*.whl /opt/tritonserver/python
    cp -r /home/debian/server/build/tritonserver/install/include/triton/core /opt/tritonserver/include/triton/core
    cp /home/debian/server/LICENSE /opt/tritonserver
    cp /home/debian/server/TRITON_VERSION /opt/tritonserver
    cd 
    git clone https://github.com/triton-inference-server/python_backend -b r24.05
    cd python_backend
    mkdir build
    # IMPORTANT : remove -WError on the two folowing lines
    # ./CMakeLists.txt:258:    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
    # ./CMakeLists.txt:266:    -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
    cd build
    export LDFLAGS="-rdynamic"
    cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r24.05 -DTRITON_COMMON_REPO_TAG=r24.05 -DTRITON_CORE_REPO_TAG=r24.05 -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=/usr/local/bin/python3.10 ..
    make install
    sudo apt install libevent_dev
    # let's test it
    # add a demo python model
    cd
    sudo mkdir -p /models/add_sub/1/
    sudo cp python_backend/examples/add_sub/model.py /models/add_sub/1/model.py
    sudo cp python_backend/examples/add_sub/config.pbtxt /models/add_sub/config.pbtxt
    python3 -m pip install tritonclient[all]
    # run triton
    /opt/tritonserver/bin/tritonserver --model-repository=/models
    # in another terminal test our python model
    cd python_backend
    python3 examples/add_sub/client.py
    # add to bashrc
    export PYTHONNOUSERSITE=True

## Packaging an env
Create a new env : 
  - export PYTHONNOUSERSITE=True
  - # create the env following above method
  - # test it
  - conda install -c conda-forge libstdcxx-ng=12 -y
  - conda install conda-pack
  - conda-pack
  - mettre l'archive obtenue dans ./models/sentence_trad/ (ou à l'endroit ou votre model l'attendra)

deploy the new env
  La suite est maintenant faite par le script deploy_triton en précisant deploy_env :
  - cp traduction_env.tar.gz /models/sentence_trad/traduction_env.tar.gz
  - /models/sentence_trad/traduction_env
  - tar -xvf /models/sentence_trad/traduction_env.tar.gz -C /models/sentence_trad/traduction_env
  - for langchain, emacs /models/sentence_trad/traduction_env/lib/python3.10/site-packages/pydantic/_internal/_typing_extra.py
    - et ajouter typeerror en exception a la method eval_type_lenient (l258):
          except (NameError, TypeError):

Export onnx:
  #pip install optimum[exporters]
  #python -m pip install --upgrade onnxruntime-gpu
  python -m pip install onnxruntime-gpu==1.18.1
  python -m pip install optimum[onnxruntime-gpu]

  optimum-cli export onnx --model Unbabel/TowerInstruct-Mistral-7B-v0.2  tower_onnx/ --task text-generation
  
  test_onnx.py

Install onnx backend [WIP]:
- Currently it fails.

    cd
    https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    git checkout v1.18.0
    # modify python version in build.sh to get python 3.10
    ./build.sh --use_cuda --cudnn_home /usr/local/cuda-12 --cuda_home <cuda home path>
    git clone https://github.com/triton-inference-server/onnxruntime_backend.git
   cd onnxruntime_backend
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/home/debian/onnxruntime/build/Linux/Debug/ -DTRITON_ONNXRUNTIME_LIB_PATHS=/home/debian/onnxruntime/build/Linux/Debug/ -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.18.0 -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=/usr/local/bin/python3.10 ..
    

Tensorrt install:
  git clone https://github.com/triton-inference-server/tensorrt_backend.git
  git checkout r24.06
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=/usr/local/bin/python3.10 -DNVINFER_LIBRARY=/home/debian/TensorRT-10.0.1.6/lib/libnvinfer.so -DNVINFER_PLUGIN_LIBRARY=/home/debian/TensorRT-10.0.1.6/lib/libnvinfer_plugin.so -DTRITON_TENSORRT_INCLUDE_PATHS=/home/debian/TensorRT-10.0.1.6/include ..
  make install

Conversion to tensort:
  cd TensorRT-10.0.1.6/bin 
  chmod +x trtexec  , 
  cd.. 
  export PATH=$PATH:/home/debian/TensorRT-10.0.1.6/bin   
  echo 'export PATH=$PATH:/home/debian/TensorRT-10.0.1.6/bin' >> ~/.bashrc
  source ~/.bashrc
   
  Conversion :
   
  trtexec --onnx=tower_onnx/model.onnx --saveEngine=model_tower.trt --fp16

Conversion:
  python -m pip install torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.nvidia.com tensorrt-cu12-libs
  pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
  python -m pip install onnxruntime-gpu==1.18.1
  python -m pip install optimum[onnxruntime-gpu]

  optimum-cli export onnx --model Unbabel/TowerInstruct-Mistral-7B-v0.2  tower_onnx_2/ --task text-generation
  trtexec --onnx=tower_onnx_2/model.onnx --saveEngine=model_tower.trt --fp16
  

Tensorrt llm:
  git clone https://github.com/NVIDIA/TensorRT-LLM.git
  cd TensorRT-LLM/
  sudo apt install libmpich-dev
  sudo apt install libopenmpi-dev
  sudo apt-get install git-lfs
  git lfs pull
  # get os angostic installer from https://developer.nvidia.com/nccl/nccl-legacy-downloads v2.21.5
  # scp ..\..\Downloads\nccl_2.21.5-1+cuda12.4_x86_64.txz debian@91.134.30.63:~/TensorRT-LLM
  xz -d nccl_2.21.5-1+cuda12.4_x86_64.txz
  tar xvf nccl_2.21.5-1+cuda12.4_x86_64.tar
  
  sudo python3.10 ./scripts/build_wheel.py --trt_root /home/debian/TensorRT-10.0.1.6/ -D "CMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc" -D "Python_EXECUTABLE=/usr/local/bin/python3.10" -D"PYBIND11_FINDPYTHON=ON" "-DPython3_EXECUTABLE=/usr/local/bin/python3.10" --clean
  