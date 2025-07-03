PROXY_URL="http://192.168.4.6:8080"

export http_proxy=$PROXY_URL
export https_proxy=$PROXY_URL
export HTTP_PROXY=$PROXY_URL
export HTTPS_PROXY=$PROXY_URL

# Build Docker image with proxy settings
docker build \
  --build-arg HTTP_PROXY=$PROXY_URL \
  --build-arg HTTPS_PROXY=$PROXY_URL \
  --build-arg http_proxy=$PROXY_URL \
  --build-arg https_proxy=$PROXY_URL \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t ik-example .