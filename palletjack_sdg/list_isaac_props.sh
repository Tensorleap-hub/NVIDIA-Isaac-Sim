#!/bin/bash
# List Isaac Sim Nucleus assets recursively.
# Usage:
#   ./list_isaac_props.sh
#   ./list_isaac_props.sh "omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props"

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
SEARCH_ROOT="${1:-$ISAAC_SIM_PATH}"
PATTERN="${2:-*.usd}"

echo "Searching for '$PATTERN' under: $SEARCH_ROOT"
echo ""
find "$SEARCH_ROOT" -iname "$PATTERN" | sort
