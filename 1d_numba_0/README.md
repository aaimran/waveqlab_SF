# rupture_1d runner

Usage:

```bash
python3 rupture_1d.py example_run.in
```

This script runs the 1D rupture simulation using the local modules `time_integrator.py`, `rate.py`, and `utils.py`.

Output: a compressed NumPy file named `<output_prefix>_domain_output.npz` containing:

- `DomainOutput_l`: (nx, nt+1, 2) — left domain velocity/stress time-history
- `DomainOutput_r`: (nx, nt+1, 2) — right domain velocity/stress time-history
- `y_l`, `y_r`: 1D spatial coordinates
- `time`: 1D time array
- `slip`, `sliprate`, `traction`: 1D time-series arrays
- `params`: JSON string with run parameters

Edit `example_run.in` to change parameters.
