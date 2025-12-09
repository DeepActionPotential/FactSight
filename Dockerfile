# 1. Use a base Python image
FROM python:3.10-slim


# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependencies file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# 5. Expose the required port (Hugging Face default)
EXPOSE 7860

# 6. Start the production server (Gunicorn is highly recommended for production)
# The format is: gunicorn -b <host>:<port> <app_file>:<flask_app_object>
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
