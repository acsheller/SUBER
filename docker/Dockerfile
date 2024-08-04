# Use an appropriate base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    rsync \
    curl \
    git \
    git-lfs \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Set PATH to include Miniconda
ENV PATH="/opt/miniconda/bin:$PATH"

# Create a non-root user named 'suber'
RUN useradd -m -s /bin/bash suber && echo "suber:suber" | chpasswd && adduser suber sudo

# Create the SUBER directory and set permissions
RUN mkdir -p /home/suber/SUBER && chown -R suber:suber /home/suber/SUBER

# Switch to the new user
USER suber
WORKDIR /home/suber/SUBER

# Copy the requirements.txt file into the container
     
COPY requirements.txt /tmp/

# Initialize conda for bash
RUN /opt/miniconda/bin/conda init bash

# Create and activate the MPR conda environment, then install necessary packages
RUN /opt/miniconda/bin/conda create -n MPR python=3.9.0 -y && \
    /opt/miniconda/bin/conda run -n MPR conda install -c nvidia/label/cuda-11.8.0 cuda -y && \
    /opt/miniconda/bin/conda run -n MPR conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y

RUN /opt/miniconda/bin/conda run -n MPR conda install ipython jupyterlab -y

# Export necessary environment variables and install Python requirements
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" > set_env_vars.sh && \
    echo "conda activate MPR" >> set_env_vars.sh && \
    echo "export TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0'" >> set_env_vars.sh && \
    echo "export GITHUB_ACTIONS=true" >> set_env_vars.sh && \
    echo "pip install -r /tmp/requirements.txt" >> set_env_vars.sh && \
    echo "pip install wikidata" >> set_env_vars.sh  # Add this line to install the wikidata module

# Make the script executable
RUN chmod +x set_env_vars.sh

# Execute the script to set up the environment
RUN /bin/bash set_env_vars.sh

# TODO add git config --global --add safe.directory /home/suber/SUBER to the dockerfile

# Create entrypoint script to activate conda environment
RUN echo '#!/bin/bash ' > /home/suber/entrypoint.sh && \
    echo 'source /opt/miniconda/etc/profile.d/conda.sh' >> /home/suber/entrypoint.sh && \
    echo 'conda activate MPR' >> /home/suber/entrypoint.sh && \
    echo 'exec "$@"' >> /home/suber/entrypoint.sh && \
    chmod +x /home/suber/entrypoint.sh

# Set entrypoint
#ENTRYPOINT ["/home/suber/entrypoint.sh"]

# Set up environment
RUN echo "conda activate MPR" >> ~/.bashrc

# Set entrypoint to bash
#ENTRYPOINT ["bash", "-l", "-c"]

EXPOSE 5003
EXPOSE 8888
# The code to run when container is started:
#ENTRYPOINT ["/home/suber/entrypoint.sh"]

# Default command
CMD ["bash"]
