# LOOK-DGC Docker Setup Guide

**Developed by: Gopichand**  
**Project: LOOK-DGC - Digital Image Forensics Toolkit**

This guide explains how to run LOOK-DGC using Docker for easy deployment and sharing.

## Prerequisites

### 1. Install Docker
- **Windows**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Install Docker using your package manager
- **macOS**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)

### 2. GUI Support Setup

#### Windows
1. Install [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)
2. Launch VcXsrv with these settings:
   - Display number: 0
   - Start no client: ✓
   - Clipboard: ✓
   - Primary Selection: ✓
   - Native opengl: ✓
   - Disable access control: ✓

#### Linux
```bash
# Allow Docker to connect to X server
xhost +local:docker
```

#### macOS
1. Install [XQuartz](https://www.xquartz.org/)
2. Start XQuartz and enable "Allow connections from network clients"
3. Run: `xhost +localhost`

## Quick Start

### Method 1: Docker Compose (Recommended)

#### Windows
```cmd
docker-compose -f docker-compose.windows.yml up
```

#### Linux/macOS
```bash
docker-compose up
```

### Method 2: Manual Docker Commands

#### Build the image
```bash
docker build -t look-dgc:latest .
```

#### Run the container

**Windows:**
```cmd
docker run -it --rm -e DISPLAY=host.docker.internal:0 -v "%cd%\images":/app/images -v "%cd%\output":/app/output look-dgc
```

**Linux:**
```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output look-dgc
```

**macOS:**
```bash
docker run -it --rm -e DISPLAY=host.docker.internal:0 -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output look-dgc
```

## Running LOOK-DGC

Once the container starts, you'll see a welcome message. To launch the GUI application:

```bash
python launch_look_dgc.py
```

## Volume Mounts

The Docker setup includes these volume mounts:
- `./images:/app/images` - For input images
- `./output:/app/output` - For analysis results

## Troubleshooting

### GUI Not Displaying

**Windows:**
- Ensure VcXsrv is running
- Check Windows Firewall settings
- Try: `docker run -it --rm -e DISPLAY=host.docker.internal:0.0 look-dgc`

**Linux:**
- Run `xhost +local:docker` before starting container
- Check DISPLAY variable: `echo $DISPLAY`
- Ensure X11 forwarding is enabled

**macOS:**
- Ensure XQuartz is running
- Check network client connections are allowed
- Try: `docker run -it --rm -e DISPLAY=docker.for.mac.localhost:0 look-dgc`

### Permission Issues
```bash
# Linux: Fix file permissions
sudo chown -R $USER:$USER ./images ./output
```

### Container Won't Start
```bash
# Check Docker is running
docker --version

# View container logs
docker logs look-dgc-app

# Remove old containers
docker container prune
```

## Building Custom Images

To modify the Docker image:

1. Edit the `Dockerfile`
2. Rebuild: `docker build -t look-dgc:custom .`
3. Run: `docker run -it --rm look-dgc:custom`

## Performance Tips

- Use local volumes for better I/O performance
- Allocate sufficient memory to Docker (4GB+ recommended)
- Close unnecessary applications when running analysis

## Security Notes

- The container runs as a non-root user for security
- X11 forwarding is required for GUI display
- Only mount necessary directories

## Support

For issues with LOOK-DGC Docker setup:
1. Check this troubleshooting guide
2. Verify Docker and X11 setup
3. Test with simple X11 applications first

---

**LOOK-DGC** - Making digital image forensics accessible to everyone.