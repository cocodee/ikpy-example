# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR /opt/conda

# Install system-level dependencies required for the project and conda
RUN apt-get update && apt-get install -y wget vim  build-essential     && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR &&  \
    rm ~/miniconda.sh &&  \
    $CONDA_DIR/bin/conda clean -tip

# Add conda to the PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the environment file to the working directory
COPY environment.yml .

# Create the conda environment from the environment.yml file
# This will create an environment named 'ik_env' as specified in the file
RUN conda env create -f environment.yml

RUN conda init bash && echo "conda activate ik_env" >> ~/.bashrc

# Set the default command to run in a bash shell with the conda environment activated
# This makes the container ready for interactive use or for running scripts
CMD ["/bin/bash", "-c", "source activate ik_env && exec /bin/bash"]
