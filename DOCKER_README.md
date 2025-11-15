# Docker Deployment Guide

## Building the Docker Image

```bash
docker build -t object-remover-backend .
```

## Running the Container

### Windows (Command Prompt)
```cmd
set FAL_KEY=your-fal-key-here
docker run -d -p 8000:8000 -e FAL_KEY=%FAL_KEY% --name object-remover object-remover-backend
```

### Linux/Mac
```bash
export FAL_KEY='your-fal-key-here'
docker run -d -p 8000:8000 -e FAL_KEY=$FAL_KEY --name object-remover object-remover-backend
```

## Accessing the Application

Once running, access the API at: `http://localhost:8000`

Health check: `http://localhost:8000/health`

## Stopping the Container

```bash
docker stop object-remover
docker rm object-remover
```

## Viewing Logs

```bash
docker logs object-remover
```

## Notes

- The FAL_KEY must be set as an environment variable before running the container
- Get your FAL API key from: https://fal.ai/dashboard/keys
- The application runs on port 8000 by default
