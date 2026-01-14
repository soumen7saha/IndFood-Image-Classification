FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY ./src /app/src
COPY ./models/food_classifier_convnexts_v2.onnx /app/models/
COPY ./models/food_classifier_convnexts_v2.onnx.data /app/models/
COPY ./models/food_resnet_v42_12_0.887.pth /app/models/
COPY ./templates /app/templates
COPY ./static /app/static

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY app.py ./
# COPY main.py src/scripts/predict.py models/model.bin ./

RUN uv sync --frozen

EXPOSE 9696

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]

# cat Dockerfile
# docker build -t indfood-imgclassification .
# docker run -it --rm -p 9696:9696 indfood-imgclassification