FROM python:3.10-slim

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY baseline_serve.py .
COPY serve .
RUN chmod +x serve

ENV PATH="/opt/program:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["serve"]
