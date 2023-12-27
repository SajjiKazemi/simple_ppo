FROM python:3.8 as base

# Install essential tools and create a non-root user for better security
ENV USER=docker
RUN apt-get update && apt-get install -y sudo curl zsh wget git \
    && rm -rf /var/lib/apt/lists/*
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Install packages for X11 forwarding
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1

# Set up a new user and configure permissions
RUN mkdir /home/${USER}/simple_ppo
WORKDIR /home/${USER}/simple_ppo
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid
    
RUN mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

# Set the working directory and install oh-my-zsh
WORKDIR /home/${USER}/simple_ppo
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install Miniconda and activate it
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

# Copy and create Conda environment based on the provided environment.yml file
COPY ppo_env.yml /home/${USER}/simple_ppo/
RUN conda env create --file ppo_env.yml

# Install OpenAI Gym dependencies
RUN apt-get install -y xvfb

# Start Xvfb (X Virtual Framebuffer) in the background
CMD ["xvfb-run", "bash"]
