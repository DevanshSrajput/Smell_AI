FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# RDKit and scikit-learn wheels need the GNU OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python src/smell_model.py && python src/train_nose.py

EXPOSE 8501 7860
HEALTHCHECK CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '8501'); urllib.request.urlopen(f'http://localhost:{p}/_stcore/health')" || exit 1
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
