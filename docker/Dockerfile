# syntax=docker/dockerfile:1.7

FROM mambaorg/micromamba:1.5.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        ca-certificates \
        curl \
        wget \
        libcairo2 \
        libfreetype6 \
        libfontconfig1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml pyproject.toml README.md LICENSE.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER api ./api

RUN mkdir -p ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh,uid=57439,gid=57439 ssh -o BatchMode=yes -T git@github.com || true

RUN --mount=type=ssh,uid=57439,gid=57439 micromamba create -y -f environment.yml \
    && micromamba clean --all --yes

WORKDIR /app
EXPOSE 8001

CMD ["micromamba", "run", "-n", "pkasso_webserver", "pkasso-web", "--host", "0.0.0.0", "--port", "8001"]
