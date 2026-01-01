# Docker Setup for TensorFlow Federated

## Quick Start

### Option 1: Using Docker Compose (Recommended - Easiest)

```bash
# Build and start the container
docker-compose up --build

# This will:
# 1. Build the Docker image with Python 3.10 + TFF
# 2. Start Jupyter Lab on http://localhost:8888
# 3. Mount your current directory as /workspace
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t federate-tff .

# Run the container
docker run -it -p 8888:8888 -v %cd%:/workspace federate-tff
```

---

## Using with VS Code

### Method 1: Connect to Docker Container (Recommended)

1. **Install VS Code Extensions**:
   - "Dev Containers" (by Microsoft)
   - "Python" (by Microsoft)

2. **Open folder in container**:
   - Press `Ctrl+Shift+P`
   - Type "Dev Containers: Open Folder in Container"
   - Select your project folder
   - VS Code will build and open the container

3. **Open your notebook and select kernel**:
   - Open `experiment.ipynb`
   - Click kernel selector (top right)
   - Select "Python 3.10"
   - Run cells - TFF will work! ✅

### Method 2: Use Jupyter Lab in Browser

1. **Start the container**:
   ```bash
   docker-compose up
   ```

2. **Open Jupyter Lab**:
   - Navigate to `http://localhost:8888`
   - You'll see Jupyter Lab interface

3. **Open experiment.ipynb** from Jupyter Lab file explorer

---

## Verification

To verify TFF is installed, run this in Docker:

```bash
docker-compose exec federate-tff python -c "import tensorflow_federated as tff; print(f'✅ TensorFlow Federated {tff.__version__}')"
```

---

## Stopping the Container

```bash
# Stop gracefully
docker-compose down

# Or kill it
docker-compose kill
```

---

## Useful Commands

```bash
# Enter container shell
docker-compose exec federate-tff bash

# View logs
docker-compose logs -f

# Rebuild image (if Dockerfile changes)
docker-compose up --build --force-recreate

# Remove everything
docker-compose down -v
```

---

## Environment Details

- **Python**: 3.10
- **TensorFlow**: 2.15.0
- **TensorFlow Federated**: Latest stable version
- **Jupyter**: Lab + Notebook
- **Port**: 8888

All your project files are mounted as `/workspace` inside the container.
