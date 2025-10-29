#!/bin/bash

# Launch the pod and capture the output
echo "Creating pod..."
POD_OUTPUT=$(runpodctl create pod \
  --name "ml-job-1" \
  --gpuType "NVIDIA GeForce RTX 5090" \
  --gpuCount 1 \
  --mem 117 \
  --vcpu 15 \
  --containerDiskSize 30 \
  --volumeSize 50 \
  --volumePath "/workspace" \
  --ports "8888/http" \
  --ports "22/tcp" \
  --startSSH \
  --secureCloud \
  --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")

echo "Pod creation output:"
echo "$POD_OUTPUT"

# Extract pod ID from the output
# Try different common patterns for pod ID extraction
POD_ID=$(echo "$POD_OUTPUT" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -z "$POD_ID" ]; then
    POD_ID=$(echo "$POD_OUTPUT" | grep -o 'pod "[^"]*"' | sed 's/pod "\([^"]*\)"/\1/')
fi
if [ -z "$POD_ID" ]; then
    POD_ID=$(echo "$POD_OUTPUT" | grep -o 'pod-[a-f0-9-]*' | head -1)
fi
if [ -z "$POD_ID" ]; then
    POD_ID=$(echo "$POD_OUTPUT" | grep -o '[a-f0-9]\{8\}-[a-f0-9]\{4\}-[a-f0-9]\{4\}-[a-f0-9]\{4\}-[a-f0-9]\{12\}' | head -1)
fi
if [ -z "$POD_ID" ]; then
    POD_ID=$(echo "$POD_OUTPUT" | grep -o '[a-f0-9]\{24\}' | head -1)
fi

if [ -z "$POD_ID" ]; then
    echo "Error: Could not automatically extract pod ID from output."
    echo "Please manually enter the pod ID:"
    read POD_ID
else
    echo "Automatically extracted pod ID: $POD_ID"
fi

# Wait for the pod to be ready
echo "Waiting for pod to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    echo "Checking pod status (attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)..."
    SSH_CMD=$(runpodctl ssh connect $POD_ID)
    echo "SSH command: $SSH_CMD"
    
    if echo "$SSH_CMD" | grep -q "not yet ready"; then
        echo "Pod not ready yet, waiting 10 seconds..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    else
        echo "Pod is ready!"
        break
    fi
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "Error: Pod did not become ready within $((MAX_ATTEMPTS * 10)) seconds"
    exit 1
fi

# Extract SSH connection details including port
SSH_USER=$(echo "$SSH_CMD" | grep -o 'ssh [^@]*@' | sed 's/ssh //' | sed 's/@//')
SSH_HOST=$(echo "$SSH_CMD" | grep -o '@[^ ]*' | sed 's/@//')
SSH_PORT=$(echo "$SSH_CMD" | grep -o '\-p [0-9]*' | sed 's/-p //')

if [ -z "$SSH_USER" ] || [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "Error: Could not extract SSH connection details from: '$SSH_CMD'"
    echo "Please check the pod status and try again."
    exit 1
fi

echo "SSH User: $SSH_USER"
echo "SSH Host: $SSH_HOST"
echo "SSH Port: $SSH_PORT"

# Upload SSH keys using SSH (since SCP requires authentication)
if [ -f "$HOME/.ssh/id_ed25519_github_external.pub" ]; then
    echo "Uploading SSH public key..."
    cat $HOME/.ssh/id_ed25519_github_external.pub | ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST "cat > ~/id_ed25519_github_external.pub"
fi

if [ -f "$HOME/.ssh/id_ed25519_github_external" ]; then
    echo "Uploading SSH private key..."
    cat $HOME/.ssh/id_ed25519_github_external | ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST "cat > ~/id_ed25519_github_external"
fi

# Set up SSH directory and keys using SSH
echo "Setting up SSH directory and keys..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p ~/.ssh && chmod 700 ~/.ssh"

# Move uploaded files to SSH directory
ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST "
    mv ~/id_ed25519_github_external.pub ~/.ssh/ 2>/dev/null || true
    mv ~/id_ed25519_github_external ~/.ssh/ 2>/dev/null || true
    chmod 644 ~/.ssh/id_ed25519_github_external.pub 2>/dev/null || true
    chmod 600 ~/.ssh/id_ed25519_github_external 2>/dev/null || true
"

# Set up SSH agent and clone the GitHub repository
echo "Setting up SSH agent and cloning GitHub repository..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST "
    eval \$(ssh-agent -s) && \
    ssh-add ~/.ssh/id_ed25519_github_external && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    git clone git@github.com:aemartinez/playground-llm-training.git /workspace/playground-llm-training
"

echo "Script execution completed. You can connect to the pod with: runpodctl connect $POD_ID"
