FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements-app.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser \
    outputs/svm/rbf_svc_direct_model.joblib \
    outputs/svm/linearsvc_direct_model.joblib \
    ./outputs/svm/
COPY --chown=appuser:appuser \
    outputs/logistic_regression/lr_padding_model.joblib \
    ./outputs/logistic_regression/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
