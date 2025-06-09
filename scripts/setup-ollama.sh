#!/bin/bash

# Script to download the llama3.1:8b model in Ollama container
# Run this after starting the containers with docker-compose up -d

echo "Downloading llama3.1:8b model in Ollama container..."
docker-compose exec ollama ollama pull llama3.1:8b

echo "Model downloaded successfully!"
echo "You can now use the RAG API with Ollama."