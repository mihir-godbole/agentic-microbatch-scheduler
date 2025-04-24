FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-venv git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy source
COPY . /workspace
WORKDIR /workspace

# Build C++/CUDA components (placeholder)
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

EXPOSE 8080 9090

ENTRYPOINT ["python", "-m", "scheduler.request_collector"]
