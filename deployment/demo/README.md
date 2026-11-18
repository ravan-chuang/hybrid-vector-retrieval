# Portable local demo

From the repository root, start the demo with Docker Compose.

```bash
docker compose -f deployment/demo/compose.yaml up --build --wait --wait-timeout 300
```

Open [the dashboard](http://127.0.0.1:5176/), add a document and search **Application · Managed Documents**. The API is available under `/api/` on the same address. To choose another local port, set `DEMO_PORT` before running Compose.

Docker builds the frontend and C++ binding and installs the Python libraries inside the images. The first run needs internet access for the base images, packages and the MiniLM embedding model. Rerank downloads the Cross-Encoder on first use. The stack uses a one-passage BM25 startup fixture, an empty application database and an empty benchmark table. The published port is bound to loopback for local use.

Stop the demo and remove its containers with the following command.

```bash
docker compose -f deployment/demo/compose.yaml down
```

The PostgreSQL and model-cache volumes persist across `down` and `up`. The CI verification uses a separate Compose project and checks the built frontend, document and search lifecycle, and persistence after recreating the containers.
