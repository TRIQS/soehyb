# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:unstable-ubuntu-clang
ARG APPNAME=triqs_soehyb

# Install here missing dependencies, e.g.
RUN apt-get update && apt-get install -y python3-h5py

# Install pyed
RUN git clone https://github.com/HugoStrand/pyed $SRC/pyed
ENV PYTHONPATH=$SRC/pyed:$PYTHONPATH

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_ID
ARG CMAKE_ARGS
RUN cmake $SRC/$APPNAME -DTRIQS_ROOT=${INSTALL} $CMAKE_ARGS && make -j4 || make -j1 VERBOSE=1
USER root
RUN make install
