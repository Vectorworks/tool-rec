FROM nvcr.io/nvidia/pytorch:23.06-py3

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install pip dependencies
COPY sagemaker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy application code
COPY transformers4rec/ /opt/program/transformers4rec/
COPY peft/ /opt/program/peft/
COPY model/patches.py /opt/program/model/patches.py
COPY model/__init__.py /opt/program/model/__init__.py
COPY sagemaker/ /opt/program/sagemaker/

# SageMaker expects the serve script on PATH
ENV PATH="/opt/program/sagemaker:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program

EXPOSE 8080

ENTRYPOINT []
CMD ["serve"]
