# PyCXpress

<div align="center">

[![Build status](https://github.com/chaoqing/PyCXpress/workflows/build/badge.svg?branch=master&event=push)](https://github.com/chaoqing/PyCXpress/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/PyCXpress.svg)](https://pypi.org/project/PyCXpress/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/chaoqing/PyCXpress/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/chaoqing/PyCXpress/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/chaoqing/PyCXpress/releases)
[![License](https://img.shields.io/github/license/chaoqing/PyCXpress)](https://github.com/chaoqing/PyCXpress/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

PyCXpress is a high-performance hybrid framework that seamlessly integrates Python and C++ to harness the flexibility of Python and the speed of C++ for efficient and expressive computation, particularly in the realm of deep learning and numerical computing.

</div>

## 🛡 Developer

Use following `Makefile` to install the dependencies.
```
WGET := wget --no-verbose

install-environment: install-tensorflow install-pytorch install-clang install-cmake install-tools

install-tools:
	command -v doxygen || sudo apt install doxygen

TENSORFLOW_VERSION := 2.10.1
TENSORFLOW_DEB := libtensorflow-cc_$(TENSORFLOW_VERSION)-gpu_$(shell dpkg --print-architecture).deb
install-tensorflow:
	[ -f $(THIRD_PARTY_DIR)/$(TENSORFLOW_DEB) ] || $(WGET) -O $(THIRD_PARTY_DIR)/$(TENSORFLOW_DEB) "https://github.com/ika-rwth-aachen/libtensorflow_cc/releases/download/v$(TENSORFLOW_VERSION)/$(TENSORFLOW_DEB)"
	$(call message, "sudo dpkg -i $(THIRD_PARTY_DIR)/$(TENSORFLOW_DEB) && sudo ldconfig")
	rm -rf $(THIRD_PARTY_DIR)/libtensorflow_cc && mkdir -p $(THIRD_PARTY_DIR)/libtensorflow_cc
	ar -p $(THIRD_PARTY_DIR)/$(TENSORFLOW_DEB) data.tar.xz | tar --xz -xf - --strip-components=3 -C $(THIRD_PARTY_DIR)/libtensorflow_cc
	mkdir -p $(THIRD_PARTY_DIR)/libtensorflow_cc/include/tensorflow/third_party/gpus && ln -sfn /usr/local/cuda $(THIRD_PARTY_DIR)/libtensorflow_cc/include/tensorflow/third_party/gpus/
	$(call message, Install dependency if needed with APT "sudo apt install libcudart11.0 libcublas11 libcufft10 libcusparse11")
	$(call message, Install dependency from JetPack6.0 APT repo "https://repo.download.nvidia.com/jetson/common/pool/main/c/cudnn/libcudnn8_8.9.4.25-1+cuda12.2_arm64.deb")
	$(call message, Install tensorflow python for Jetson with "wget 'https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1%2Bnv24.08-cp310-cp310-linux_aarch64.whl'")
	$(call message, And then pip install with "python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v61 tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl")


install-pytorch:
	[ -f $(THIRD_PARTY_DIR)/libtorch.zip ] || $(WGET) -O $(THIRD_PARTY_DIR)/libtorch.zip "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
	rm -rf $(THIRD_PARTY_DIR)/libtorch && mkdir -p $(THIRD_PARTY_DIR)/libtorch
	env -C $(THIRD_PARTY_DIR) unzip $(THIRD_PARTY_DIR)/libtorch.zip

LLVM_APT_URL := $(shell . /etc/os-release && echo "http://apt.llvm.org/$${VERSION_CODENAME}/ llvm-toolchain-$${VERSION_CODENAME}")
LLVM_VERSION := 20
install-clang:
	sudo apt-get -y purge --auto-remove clang*
	$(call message,Deprecation Warnning)
	$(call message,    "$(WGET) -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -")
	$(WGET) -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/llvm-snapshot.asc
	printf "deb $(LLVM_APT_URL)%s main\n" "" "-18" "-19" | sudo tee /etc/apt/sources.list.d/llvm.list
	sudo apt-get update
	DEBIAN_FRONTEND=noninteractive sudo apt-get -y install --no-install-recommends clang-$(LLVM_VERSION) clangd-$(LLVM_VERSION) clang-format-$(LLVM_VERSION) clang-tidy-$(LLVM_VERSION) libclang-rt-$(LLVM_VERSION)-dev
	sudo ln -sfn $$(command -v clang-$(LLVM_VERSION)) /usr/bin/clang
	sudo ln -sfn $$(command -v clangd-$(LLVM_VERSION)) /usr/bin/clangd
	sudo ln -sfn $$(command -v clang++-$(LLVM_VERSION)) /usr/bin/clang++
	sudo ln -sfn $$(command -v clang-format-$(LLVM_VERSION)) /usr/bin/clang-format
	sudo ln -sfn $$(command -v clang-tidy-$(LLVM_VERSION)) /usr/bin/clang-tidy

install-cmake:
	chmod +x $(THIS_MAKEFILE_DIR)/.devcontainer/reinstall-cmake.sh
	sudo $(THIS_MAKEFILE_DIR)/.devcontainer/reinstall-cmake.sh 3.22.2

install-cmake:
  git clone https://ghp.ci/https://github.com/protocolbuffers/protobuf
  git submodule update --init --recursive
  ./autogen.sh
  ./configure --prefix=$HOME/Work/PyCXpress/third_party/protobuf
  make
  make install
```

## 🛡 License

[![License](https://img.shields.io/github/license/chaoqing/PyCXpress)](https://github.com/chaoqing/PyCXpress/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/chaoqing/PyCXpress/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{PyCXpress,
  author = {chaoqing},
  title = {PyCXpress is a high-performance hybrid framework that seamlessly integrates Python and C++ to harness the flexibility of Python and the speed of C++ for efficient and expressive computation, particularly in the realm of deep learning and numerical computing.},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chaoqing/PyCXpress}}
}
```
