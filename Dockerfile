FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Service venv (FastAPI, no torch)
RUN python -m venv /venvs/service \
 && /venvs/service/bin/pip install \
      fastapi==0.163.1 uvicorn==0.54.0 \
      numpy==2.5.4 pillow==13.0.0 PyYAML==6.0.3 psutil==7.2.2

# Embedding worker venv (CPU torch + open_clip)
RUN python -m venv /venvs/embed \
 && /venvs/embed/bin/pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.15.0 torchvision==0.29.2 \
 && /venvs/embed/bin/pip install \
      open_clip_torch==4.2.1 timm==1.0.31 huggingface_hub==2.4.0 \
      safetensors==0.8.0 numpy==2.5.4 pillow==13.0.0 PyYAML==6.0.3

# Plate worker venv (CPU torch + ultralytics)
RUN python -m venv /venvs/yolo \
 && /venvs/yolo/bin/pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.15.0 torchvision==0.29.2 \
 && /venvs/yolo/bin/pip install \
      ultralytics==8.4.202 opencv-python==5.0.0.93 \
      numpy==2.5.4 pillow==13.0.0 PyYAML==6.0.3

WORKDIR /app
# Non-editable install of the package into the service venv
COPY pyproject.toml ./
COPY src ./src
RUN /venvs/service/bin/pip install --no-deps .

COPY scripts ./scripts
COPY research ./research

EXPOSE 8000
ENTRYPOINT ["/venvs/service/bin/python", "scripts/run_service.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--embed-worker-python", "/venvs/embed/bin/python", "--plate-worker-python", "/venvs/yolo/bin/python"]
