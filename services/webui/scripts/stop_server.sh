
echo "Stopping Live VLM WebUI server..."
pkill -f "live_vlm_webui.server"

# Wait a moment
sleep 1

# Check if stopped
if pgrep -f "live_vlm_webui.server" > /dev/null; then
    echo "❌ Server still running, forcing kill..."
    pkill -9 -f "live_vlm_webui.server"
    sleep 1
fi

if ! pgrep -f "live_vlm_webui.server" > /dev/null; then
    echo "✓ Server stopped successfully"
else
    echo "❌ Failed to stop server"
    exit 1
fi
