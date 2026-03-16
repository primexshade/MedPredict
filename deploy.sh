#!/usr/bin/env zsh
# deploy.sh — Start the full MedPredict stack with Docker Compose
# Run from the project root: bash deploy.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

echo "🚀 MedPredict Deployment Script"
echo "================================"

# Detect Docker socket
if [ -S /var/run/docker.sock ]; then
  export DOCKER_HOST="unix:///var/run/docker.sock"
elif [ -S "$HOME/.docker/run/docker.sock" ]; then
  export DOCKER_HOST="unix://$HOME/.docker/run/docker.sock"
else
  echo "❌ Docker socket not found. Is Docker Desktop running?"
  exit 1
fi

echo "✓ Using Docker socket: $DOCKER_HOST"

# Verify Docker is reachable
docker info > /dev/null 2>&1 || { echo "❌ Cannot connect to Docker daemon"; exit 1; }
echo "✓ Docker daemon is running"

# Stop any existing containers
echo ""
echo "⏹  Stopping existing containers (if any)..."
DOCKER_HOST="$DOCKER_HOST" docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true

# Build and start all services
echo ""
echo "🔨 Building and starting all services..."
echo "   This will take 2-5 minutes on first run (needs to pull base images & install deps)"
echo ""
DOCKER_HOST="$DOCKER_HOST" docker compose -f "$COMPOSE_FILE" up -d --build

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📍 Service URLs:"
echo "   Frontend :  http://localhost:5173"
echo "   API      :  http://localhost:8000"
echo "   API Docs :  http://localhost:8000/docs"
echo "   Health   :  http://localhost:8000/health"
echo "   PostgreSQL: localhost:5432"
echo "   Redis    :  localhost:6379"
echo ""
echo "🔑 Login Credentials: admin@example.com / admin"
echo ""
echo "📋 View logs: docker compose -f docker/docker-compose.yml logs -f"
echo "⏹  Stop all:  docker compose -f docker/docker-compose.yml down"
