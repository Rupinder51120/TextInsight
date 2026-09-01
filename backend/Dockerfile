# docs/TECH_STACK.md "Deployment: Docker + Docker Compose" — Python base image, installs requirements.txt,
# runs uvicorn. Built from the repo root (see docker-compose.yml's build.context) so it can COPY the whole
# application, not just backend/.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # hf-xet (Hugging Face's newer fast-transfer backend) is known to hang model downloads in some
    # sandboxed/restricted network environments (hit during local development — see project history);
    # uninstalling it makes huggingface_hub fall back to plain HTTP downloads, which are slower but
    # reliable everywhere.
    && pip uninstall -y hf-xet || true

# Same workaround the local dev environment needed for a macOS FAISS/PyTorch OpenMP conflict at process
# exit — harmless elsewhere, and this image may run in comparable conditions.
ENV KMP_DUPLICATE_LIB_OK=TRUE

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
