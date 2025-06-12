#!/bin/bash

# Script to download the specified model in Ollama container
# Run this after starting the containers with docker-compose up -d
# Prompt user for model number with default value
read -p "Enter model number (default: llama3.1:8b): " MODEL
MODEL=${MODEL:-llama3.1:8b}

echo "Selected model: $MODEL"

echo "Downloading $MODEL model in Ollama container..."
docker-compose exec ollama ollama pull $MODEL

echo "Model downloaded successfully!"
echo "You can now use the RAG API with Ollama."