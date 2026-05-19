#!/bin/bash
# List Isaac Sim Nucleus assets recursively.
# Usage:
#   ./list_isaac_props.sh
#   ./list_isaac_props.sh "omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props"

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/opt/IsaacSim}"
ROOT="${1:-omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props}"

NVJITLINK_LIB_DIR="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"
if [ -d "$NVJITLINK_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$NVJITLINK_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$ISAAC_SIM_PATH"

./python.sh - << PYEOF
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import omni.client

ROOT = "$ROOT"

def walk(path, indent=0):
    result, entries = omni.client.list(path)
    if result != omni.client.Result.OK:
        print(f"{'  ' * indent}[ERROR {result}] {path}")
        return
    for e in entries:
        print("  " * indent + e.relative_path)
        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            walk(path + "/" + e.relative_path, indent + 1)

walk(ROOT)
simulation_app.close()
PYEOF
