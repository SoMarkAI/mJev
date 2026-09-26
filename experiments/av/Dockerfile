FROM vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089
RUN python3 -m pip install --retries 0 --index-url https://pypi.org/simple \
    'qwen-omni-utils[decord]==0.0.9' 'librosa==0.11.0' 'audioread==3.0.1' \
    'av==18.1.0' 'soundfile==0.14.0' 'pyarrow==25.0.1'
WORKDIR /app
COPY . /app
RUN python3 -m pip install --no-deps --no-build-isolation .
# pytest is the single test entry point; keep test dependencies explicit.
RUN python3 -m pip install --retries 0 "pytest==8.4.2" "accelerate==1.15.0" "requests>=2.31"
ENV PYTHONPATH=/app VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ENTRYPOINT ["python3", "-m", "mjev.benchmark.pilot"]
