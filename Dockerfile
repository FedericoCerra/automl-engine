FROM python:3.10-slim

# Hugging Face Security Requirement: Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set up the working directory and environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first (This makes future Docker builds much faster)
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY --chown=user . .

# Create the folders needed for the API and guarantee write permissions
RUN mkdir -p api_uploads api_models

# Expose the standard Hugging Face port
EXPOSE 7860

# Boot up the server 
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]