# Optional vLLM demo image. The default HF workflow does not require Docker.
FROM vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089
WORKDIR /app
COPY . /app
RUN python3 -m pip install --no-deps --no-build-isolation .
# pytest is the single test entry point; keep test dependencies explicit.
RUN python3 -m pip install --retries 0 "pytest==8.4.2" "accelerate==1.15.0" "requests>=2.31"
ENV PYTHONPATH=/app VLLM_USE_V2_MODEL_RUNNER=0
ENTRYPOINT ["python3", "demo.py"]
