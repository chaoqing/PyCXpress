ARG IMAGE_VARIANT=ubuntu-24.04
FROM mcr.microsoft.com/devcontainers/cpp:1-${IMAGE_VARIANT}

ARG TENSORFLOW_VERSION=2.13.0
ARG INSTALL_LLVM=true
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE=none

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ncdu \
    unar \
    doxygen \
    build-essential

# [Optional] Uncomment this section to install libtensorflow_cc.
RUN if [ -n "${TENSORFLOW_VERSION}" ]; then \
        wget -O /tmp/libtensorflow-cc.deb "https://github.com/ika-rwth-aachen/libtensorflow_cc/releases/download/v${TENSORFLOW_VERSION}/libtensorflow-cc_${TENSORFLOW_VERSION}-gpu_$(dpkg --print-architecture).deb" && \
        dpkg -i /tmp/libtensorflow-cc.deb && rm -f /tmp/libtensorflow-cc.deb; \
    fi \
    && ldconfig

# Optionally install the LLVM clang
RUN if [ "${INSTALL_LLVM:-false}" = "true" ]; then \
        apt-get -y purge --auto-remove clang* && \
        wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - && \
        . /etc/os-release && echo "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME} main" > /etc/apt/sources.list.d/llvm.list \
        && apt-get update && export DEBIAN_FRONTEND=noninteractive \
        && apt-get -y install --no-install-recommends clang clang-format ; \
    fi

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh
