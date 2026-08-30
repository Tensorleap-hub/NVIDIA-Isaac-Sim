from dataclasses import dataclass, field
from typing import Any, Dict


# Local stand-ins for the engine contract dataclasses referenced by engine_loop.py
# (src_tensorleap.contract.workersynthetic.request.syntheticjobrequest.SingleSimulationData
# and src_tensorleap.contract.code_loader_contract_shared.responsedataclasses.SimulationInstance),
# so the vendored loop runs without the engine repo on the path.


@dataclass
class SingleSimulationData:
    sim_name: str
    params: Dict[str, Any]
    n_samples: int
    seed: int


@dataclass
class SimulationInstance:
    name: str
    sim_config: Dict[str, Any] = field(default_factory=dict)
