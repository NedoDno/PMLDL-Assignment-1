# Use official Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /

# Copy the requirements file
COPY code/deployment/app/requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY code/deployment/app/app.py app.py

# Expose the Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
