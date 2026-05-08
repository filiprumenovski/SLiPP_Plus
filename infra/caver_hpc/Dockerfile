FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        openjdk-17-jre-headless \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slipp_plus

# Install uv once, then create the project virtualenv from locked deps.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY tools/caver ./tools/caver
COPY infra/caver_hpc ./infra/caver_hpc

RUN uv sync --locked --no-dev

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="/opt/slipp_plus/.venv/bin:/usr/lib/jvm/java-17-openjdk-amd64/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/slipp_plus/infra/caver_hpc/entrypoint.sh"]
CMD ["python", "-m", "slipp_plus.tunnel_features", "--help"]
