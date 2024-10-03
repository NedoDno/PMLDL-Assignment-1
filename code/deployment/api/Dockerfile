# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /

# Copy the requirements file to the working directory
COPY code/deployment/api/requirements.txt requirements.txt

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents (FastAPI app) into the container at /app
COPY code/deployment/api /
# Copy the model into the container
COPY code/models code/models
COPY code/__init__.py code/

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
