FROM nvcr.io/nvidia/pytorch:20.12-py3

# install requirement
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110