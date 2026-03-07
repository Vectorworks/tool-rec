FROM python:3.11-slim

# Install CPU-only PyTorch first to avoid pulling in multi-GB CUDA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY sagemaker_retrieval/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Pre-download the sentence-transformer model so it's baked into the image
# (SageMaker containers may not have internet access at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY sagemaker_retrieval/ /opt/program/sagemaker_retrieval/

ENV PATH="/opt/program/sagemaker_retrieval:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program

EXPOSE 8080

ENTRYPOINT []
CMD ["serve"]
