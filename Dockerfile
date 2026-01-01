FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install TensorFlow and compatible TensorFlow Federated
RUN pip install tensorflow==2.14.1 tensorflow-federated==0.73.0

# Install Jupyter and dependencies
RUN pip install \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    scipy \
    seaborn \
    tqdm

# Create a non-root user
RUN useradd -m -s /bin/bash jupyter

# Set up Jupyter config directory
RUN mkdir -p /home/jupyter/.jupyter && chown -R jupyter:jupyter /home/jupyter/.jupyter

# Switch to jupyter user
USER jupyter

# Set up Jupyter
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /home/jupyter/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /home/jupyter/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
