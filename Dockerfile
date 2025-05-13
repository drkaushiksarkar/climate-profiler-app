# Use an official Miniconda3 image as a base
FROM continuumio/miniconda3:4.12.0

# Install Debian build-essential (gcc/g++) so pip can compile wheels like phik & wordcloud
RUN apt-get update \
 && apt-get install -y build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# --- Create conda env & install native deps (eccodes, cfgrib, etc.) ---
RUN conda create -n climhealth python=3.10 -y && \
    echo "conda activate climhealth" >> ~/.bashrc && \
    conda install -n climhealth -c conda-forge \
        cmake eccodes cfgrib xarray \
    -y && \
    conda clean -afy

# --- Install Python dependencies via pip ---
# Copy the requirements file first (leverages Docker layer cache)
COPY requirements.txt .

# Use the pip inside our conda env to install everything, now that gcc/g++ exist
RUN /opt/conda/envs/climhealth/bin/pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
COPY app.py .
# (Uncomment if you have other modules/folders to include)
# COPY utils/ ./utils/

# --- Expose Port ---
EXPOSE 8501

# --- Entrypoint ---
CMD ["/opt/conda/envs/climhealth/bin/streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

# --- Metadata ---
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Streamlit app for Climate & Health Data Profiling"