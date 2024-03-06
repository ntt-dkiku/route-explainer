FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN echo "Building docker image"

RUN apt-get -y update && \
    apt-get install -y \
    curl \
    build-essential \
    git \
    vim \
    tmux 

# jupyter-notebook & lab
RUN python3 -m pip install jupyter
RUN python3 -m pip install jupyterlab

# LLMs
RUN python3 -m pip install openai
RUN python3 -m pip install tiktoken
RUN python3 -m pip install langchain

# Web app
RUN python3 -m pip install streamlit
RUN python3 -m pip install streamlit-folium
RUN python3 -m pip install folium

# Google Map API
RUN python3 -m pip install googlemaps

# OR-tools
RUN python3 -m pip install ortools

# other convenient packages
RUN python3 -m pip install torchmetrics
RUN python3 -m pip install scipy
RUN python3 -m pip install pandas
RUN python3 -m pip install matplotlib
RUN python3 -m pip install tqdm