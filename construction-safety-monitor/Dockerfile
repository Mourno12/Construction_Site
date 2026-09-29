# EdgeGuard - free-tier deploy image (Demo Mode only - see
# python/requirements-deploy.txt for why the full YOLO/ViT stack is
# deliberately left out). Node serves the app and REST/WebSocket API;
# it spawns the Python inference worker as a subprocess.
FROM node:20-bookworm-slim

# libgl1/libglib2.0-0: opencv-python-headless still links against these at
# import time even without GUI support.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY python/requirements-deploy.txt python/requirements-deploy.txt
RUN pip3 install --no-cache-dir --break-system-packages -r python/requirements-deploy.txt

COPY server ./server
COPY public ./public
COPY python ./python

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server/index.js"]
