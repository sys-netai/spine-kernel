

# DRL Congestion Control Framework — Spine

Spine is a Deep Reinforcement Learning (DRL)-based congestion control (CC) framework that leverages a custom Linux kernel module and simulation environments to train and evaluate networking policies.

---

## Table of Contents

- [Kernel Setup](#kernel-setup)
- [Third-Party Dependencies](#third-party-dependencies)
- [Build Environment](#build-environment)
  - [GCC](#gcc)
  - [CMake](#cmake)
  - [Protobuf](#protobuf)
  - [DI-Engine](#di-engine)
  - [OpenSSL](#openssl)
- [Compiling Spine](#compiling-spine)
- [Loading the Spine Kernel](#loading-the-spine-kernel)
- [Training](#training)
- [Inference](#inference)
  - [Simulated Inference](#simulated-inference)
  - [Real Environment Inference](#real-environment-inference)

---

## Kernel Setup

```bash
git clone git@github.com:sys-netai/astraea-kernel.git

cd ~/astraea/third_party/astraea-kernel/testbed
sudo dpkg -i linux-image*
sudo dpkg -i linux-header*
sudo reboot
```

## Third-Party Dependencies

```bash
git clone git@github.com:sys-netai/spine_private.git

cd spine_private
git submodule init
git submodule update
git submodule sync --recursive
```

------

## Build Environment

### GCC

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 1070
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 1071
```

### CMake (v3.21.1)

```bash
sudo apt remove --purge cmake
sudo apt update
sudo apt install -y build-essential libssl-dev

wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz
tar -zxvf cmake-3.21.1.tar.gz
cd cmake-3.21.1

./bootstrap
make -j$(nproc)
sudo make install

cmake --version
```

Additional dependencies:

```bash
sudo apt install -y autoconf automake libtool curl unzip
pip install markupsafe==2.0.1
```

### Protobuf

```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protobuf-all-3.20.1.tar.gz
tar -xzvf protobuf-all-3.20.1.tar.gz
cd protobuf-3.20.1

./configure
make -j$(nproc)
make check
sudo make install
sudo ldconfig
```

### DI-Engine

```bash
cd ~/spine_private/third_party/DI-engine
pip3 install -e .
```

### OpenSSL

```bash
sudo apt install -y make libssl-dev
```

------

## Compiling Spine

```bash
conda activate di-engine
cd ~/spine_private/tools/
./setup.sh
```

> After each reboot, reconfigure the kernel parameters:

```bash
sudo ./setup_kernel.sh
```

------

## Loading the Spine Kernel——neo

```bash
cd ~/spine_private/third_party/spine-kernel/src
make
sudo ./spine_kernel_load.sh
```

Verify that the `neo` congestion control algorithm is installed:

```bash
sysctl net.ipv4.tcp_available_congestion_control
```

------

## Training

1. Modify the DRL state/action/reward design in:

   ```bash
   src/agent/definitions.py
   ```

2. Start the Spine training service:

   ```bash
   cd ~/spine_private/third_party/spine-kernel/python/src/
   sudo python3 spine.py -u $USER -a neo
   ```

3. Run the training script:

   ```bash
   cd ~/spine_private/src/train/config/serial
   python se_r2d2_config.py
   ```

   - `train_world_file` and `eval_world_file` specify training and evaluation simulation environments.
   - `exp_name` determines where logs and checkpoints are saved.

4. Monitor progress using TensorBoard:

   ```bash
   tensorboard --logdir <exp_name> --port 8889
   ```

------

## Inference

### Simulated Inference

1. Start the inference service:

   ```bash
   cd ~/spine_private/third_party/spine-kernel/python/src/
   sudo python3 spine_eval.py -u $USER -a neo
   ```

2. Set `model_path` in:

   ```bash
   src/eval/neo_infer.py
   ```

3. Run evaluation:

   ```bash
   cd ~/spine_private/evaluation/eval_performance
   python eval_performance.py standard_eval.json
   ```

   - `standard_eval.json` defines the simulated environment.
   - Results are saved in `evaluation/emulation/`.

### Real Environment Inference

1. Start the inference service:

   ```bash
   cd ~/spine_private/third_party/spine-kernel/python/src/
   sudo python3 spine_eval.py -u $USER -a neo
   ```

2. Build the Spine helper:

   ```bash
   cd ~/spine_private/third_party/spine-open-source/src
   mkdir build && cd build
   cmake ..
   make -j
   ```

3. Start the Spine server:

   ```bash
   ./bin/server --port=12345
   ```

4. Run the client:

   ```bash
   ./bin/client_spine \
   --port=12345 \
   --ip=127.0.0.1 \
   --cong=neo \
   --pyhelper=../../python/eval/neo_infer.py \  # Set model_path in neo_infer.py
   --model=../../python/model/current/ckpt/model.tar \ # Path to the model checkpoint
   --interval=30
   ```