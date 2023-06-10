# FROM nvcr.io/nvidia/pytorch:23.04-py3
FROM nvcr.io/nvidia/pytorch:21.02-py3
# FROM cnstark/pytorch:1.7.1-py3.9.12

RUN apt-get update && apt-get install -y \
    lsof

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
