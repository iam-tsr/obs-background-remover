echo "====================================="
echo "Virtual Camera Setup for OBS"
echo "====================================="
echo ""

# Check if v4l2loopback package is installed
echo "Checking v4l2loopback installation..."
if ! modinfo v4l2loopback &> /dev/null; then
    echo "✗ v4l2loopback is not installed on the system."
    read -p "Do you want to install v4l2loopback now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing v4l2loopback for Fedora..."
        sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
        sudo dnf install v4l2loopback
        
    else
        echo "Please install v4l2loopback manually, then run this script again."
        exit 1
    fi
else
    echo "✓ v4l2loopback is installed"
fi

echo ""
echo "====================================="
echo "Install Dependencies"
echo "====================================="
echo ""

# Prompt user for installation method
echo "How you want to install dependencies? (virtualenv(v)/global(g))"
read -p "Enter your choice: " choice
if [[ $choice == "virtualenv" || $choice == "v" ]]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies in virtual environment..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully."

elif [[ $choice == "global" || $choice == "g" ]]; then
    echo "Installing dependencies globally..."
    sudo pip install -r requirements.txt
    echo "Dependencies installed successfully."

else
    echo "Invalid choice. Please run the script again and choose either 'virtualenv' or 'global'."
    exit 1
fi

# User Guide
echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To start the virtual camera, run:"
echo "  ./virtual_camera.py --preview"
echo ""
echo "Once running, you can add it as a Video Capture Device in OBS:"
echo "  1. Open OBS"
echo "  2. Add Source → Video Capture Device"
echo "  3. Select 'Background Blur Cam' or '/dev/video10'"
echo ""
echo "Command line options:"
echo "  --input N       Use /dev/videoN as input (default: 0)"
echo "  --blur N        Set blur strength 5-101 (default: 55)"
echo "  --preview       Show preview window"
echo "  --width N       Set output width (default: 640)"
echo "  --height N      Set output height (default: 480)"
echo "  --fps N         Set output FPS (default: 30)"
echo ""
echo "Example:"
echo "  ./virtual_camera.py --input 0 --blur 75"
echo ""