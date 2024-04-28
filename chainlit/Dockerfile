# Use an official Python runtime as a parent image
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the container
WORKDIR $HOME/app

# Copy the current directory contents into the container at /app
COPY --chown=user . $HOME/app
COPY ./requirements.txt $HOME/app/requirements.txt

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Command to run the app
CMD python -m chainlit run qa.py -h --host 0.0.0.0 --port ${PORT}