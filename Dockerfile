FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flower port
EXPOSE 8080

# Run server with environment variables
CMD python server_generic.py \
    --config ${FL_CONFIG:-configs/mnist_config.py} \
    --server-address 0.0.0.0:${PORT:-8080} \
    --num-rounds ${NUM_ROUNDS:-5} \
    --alpha ${ALPHA:-0.5} \
    --min-clients ${MIN_CLIENTS:-2}
