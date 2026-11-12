# Quasi-Static Aeroelastic Modeling and Design Assessment of Membrane-Frame Flapping Wings

Code for the study of a quasi-static aeroelastic model of membrane-frame flapping wings and its
use for wing design. This repository holds code only; the measurement tables (`measurements/`)
and raw acquisitions (`data/raw/`) are distributed separately and are required by `evaluate.py`
and `scripts/reprocess_force_power.py`.

## Layout
| path | content |
|---|---|
| `wing_lib/wingmodel/` | the model (Python port of the MATLAB original): meshes, membrane and frame energies, panel aerodynamics, coupled solver, power features, design search |
| `measurements/` | lift, electrical input and kinematics tables rebuilt from the raw acquisitions (see its README) |
| `scripts/reprocess_force_power.py` | raw acquisitions -> `measurements/*.csv` |
| `scripts/validate_port.py` | checks the Python port against the stored MATLAB outputs (36 cases) |
| `evaluate.py` | held-out evaluation: leave-one-planform-out over six reference wings plus the two designed wings |
| `program_0/` | baseline program: the model as published |
| `reference/` | MATLAB outputs and geometry used for the port validation |

## Quick start
```bash
pip install -r requirements.txt
python scripts/validate_port.py 15            # port check at 15 Hz (about 2 min on 16 cores)
python evaluate.py --program_path program_0/src/program.py --results_dir results/baseline
```

Using the model directly:
```python
import sys; sys.path.insert(0, "wing_lib")
from wingmodel import model as M, wings
mesh, material, membrane = wings.build_wing("S-Opt-K")
cycle = M.run_cycle(mesh, material, M.Motion(freq=15))
print(1e3 * cycle["mean_lift"], "mN", 1e3 * cycle["max_disp"], "mm")
rigid = M.run_cycle(mesh, material, M.Motion(freq=15), rigid=True)   # deformation-off control
```

## Port validation
Cycle-mean lift agrees with the MATLAB model to within 0.005 mN and per-phase lift to within
0.05 mN RMS over 36 cases (three planforms, two membranes, 10-15 Hz); see `reference/` and
`scripts/validate_port.py`.
