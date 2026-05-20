#!/bin/bash
# Discover Isaac Sim DigitalTwin warehouse USD assets.
# Run inside the Isaac Sim environment or on the EC2 host.
# Usage:
#   ./list_isaac_props.sh [assets_root]
#
# Searches Warehouse/Equipment, Warehouse/Vehicles, Warehouse/Shipping,
# and Environments/Simple_Warehouse/Props by default.

ASSETS_ROOT="${1:-}"

# Auto-detect if not provided
if [[ -z "$ASSETS_ROOT" ]]; then
    for candidate in \
        "/opt/IsaacSim/assets" \
        "/isaac-sim/assets" \
        "$HOME/.local/share/ov/pkg/isaac_sim-*/assets" \
        "/opt/IsaacSim"
    do
        # expand glob
        for expanded in $candidate; do
            if [[ -d "$expanded" ]]; then
                ASSETS_ROOT="$expanded"
                break 2
            fi
        done
    done
fi

if [[ -z "$ASSETS_ROOT" ]]; then
    echo "ERROR: Could not find Isaac assets root. Pass it as an argument."
    exit 1
fi

echo "Assets root: $ASSETS_ROOT"
echo ""

search_and_print() {
    local label="$1"
    local pattern="$2"
    local results
    results=$(find "$ASSETS_ROOT" -iname "*.usd" -path "$pattern" 2>/dev/null | sort)
    if [[ -n "$results" ]]; then
        echo "=== $label ==="
        echo "$results"
        echo ""
    fi
}

search_and_print "DigitalTwin / Equipment"  "*DigitalTwin*Warehouse/Equipment*"
search_and_print "DigitalTwin / Vehicles"   "*DigitalTwin*Warehouse/Vehicles*"
search_and_print "DigitalTwin / Shipping"   "*DigitalTwin*Warehouse/Shipping*"
search_and_print "Simple_Warehouse / Props" "*Simple_Warehouse/Props*"
