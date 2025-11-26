#!/bin/bash
# Setup script for Qdrant on ecetesla (UWaterloo ECE cluster)
# No root access needed - runs entirely in user space

set -e

echo "=== Qdrant Setup for ecetesla ==="
echo ""

# Configuration
QDRANT_VERSION="1.12.4"
INSTALL_DIR="$HOME"
DATA_DIR="$HOME/qdrant-data"
TARBALL="qdrant-x86_64-unknown-linux-gnu.tar.gz"
DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/${TARBALL}"

# Step 1: Create directories
echo "[1/4] Creating directories..."
mkdir -p "$DATA_DIR"

# Step 2: Download Qdrant
echo "[2/4] Downloading Qdrant v${QDRANT_VERSION}..."
cd "$INSTALL_DIR"
if [ ! -f "$TARBALL" ]; then
    wget -q --show-progress "$DOWNLOAD_URL"
else
    echo "  Tarball already exists, skipping download"
fi

# Step 3: Extract
echo "[3/4] Extracting..."
tar -xzf "$TARBALL"

# Step 4: Create startup script
echo "[4/4] Creating startup script..."
cat > "$INSTALL_DIR/start_qdrant.sh" << 'EOF'
#!/bin/bash
# Start Qdrant server
cd "$(dirname "$0")"
./qdrant
EOF
chmod +x "$INSTALL_DIR/start_qdrant.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start Qdrant:"
echo "  cd ~ && ./qdrant"
echo ""
echo "Qdrant will be available at:"
echo "  REST API: http://localhost:6333"
echo "  gRPC:     localhost:6334"
echo "  Web UI:   http://localhost:6333/dashboard"
echo ""
echo "Data will be stored in: ./storage (relative to where qdrant runs)"
echo ""

