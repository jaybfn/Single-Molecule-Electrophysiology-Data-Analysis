# Use the official Python image from Docker Hub as the base image
FROM python:3.11-slim

# Set the working directory in the Docker container
WORKDIR /pynanopore

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local pynanopore directory to the working directory
COPY pynanopore/ .

# Expose the port on which Streamlit runs
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
