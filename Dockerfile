FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 curl git \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl mesa-utils-extra libxrender1

# Clone the repository early

# FIX 3: Use absolute path for WORKDIR
WORKDIR /Hunyuan3D-2

# Now clone the repository
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git .

WORKDIR /Hunyuan3D-2

RUN sed -i '56s/^/#/' requirements.txt

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt


RUN python -m pip install -e .

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Build custom_rasterizer
RUN echo "Building custom_rasterizer..." && \
    cd hy3dgen/texgen/custom_rasterizer && \
    python setup.py install && \
    cd ../..

WORKDIR /Hunyuan3D-2

# Build differentiable_renderer
RUN echo "Building differentiable_renderer..." && \
    cd hy3dgen/texgen/differentiable_renderer && \
    python setup.py install
        
RUN apt-get install -y libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/output/{images,meshes} /cache

EXPOSE 8000

CMD ["python", "-m", "app.main"]