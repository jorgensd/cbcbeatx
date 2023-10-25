FROM ghcr.io/fenics/dolfinx/dolfinx:v0.7.0

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

RUN python3 -m pip install pip setuptools --upgrade


WORKDIR /src/
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/jorgensd/cbcbeatx
RUN python3 -m pip install ./cbcbeatx[all]


# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
