FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev liblapack-dev python3-dev \
    build-essential cmake curl wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" && \
    CMDSTAN_VER=$(ls /root/.cmdstan/ | head -1) && \
    ln -sf "/root/.cmdstan/${CMDSTAN_VER}" /usr/local/lib/python3.11/site-packages/prophet/stan_model/cmdstan-2.33.1

# Prophet 1.1.5 expects a cmdstan install at stan_model/cmdstan-2.33.1. The
# `ln` above lands *inside* that directory when the wheel already ships one,
# leaving it without a makefile. Replace it with a real symlink to the cmdstan
# that cmdstanpy installed.
RUN CMDSTAN_VER=$(ls /root/.cmdstan/ | head -1) && \
    rm -rf /usr/local/lib/python3.11/site-packages/prophet/stan_model/cmdstan-2.33.1 && \
    ln -s "/root/.cmdstan/${CMDSTAN_VER}" /usr/local/lib/python3.11/site-packages/prophet/stan_model/cmdstan-2.33.1

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:create_app()"]
