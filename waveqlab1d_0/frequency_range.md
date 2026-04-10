# Resolvable Frequency Range by Resolution

**Wave speed used:** $c_{s,min} = 3.000$ km/s (slowest layer, test-4)  
**Stencil:** 6th-order SBP-SAT  
**Practical accuracy:** PPW = 10 (points per wavelength)  
**Upper bound:** PPW = 5 (~1% phase error)

| Resolution | Δx (km) | f_min | f_max practical (PPW=10) | f_max upper (PPW=5) | f_Nyquist |
|:----------:|:-------:|------:|-------------------------:|--------------------:|----------:|
| 80m  | 0.080 | 0.125 Hz |   3.75 Hz |   7.5 Hz |  18.75 Hz |
| 40m  | 0.040 | 0.125 Hz |   7.5 Hz  |  15 Hz   |  37.5 Hz  |
| 20m  | 0.020 | 0.125 Hz |  15 Hz    |  30 Hz   |  75 Hz    |
| 10m  | 0.010 | 0.125 Hz |  30 Hz    |  60 Hz   | 150 Hz    |
| 5m   | 0.005 | 0.125 Hz |  60 Hz    | 120 Hz   | 300 Hz    |
| 1m   | 0.001 | 0.125 Hz | 300 Hz    | 600 Hz   | 1500 Hz   |

## Notes

- $f_{min} = 1/t_{end}$: tests 1–3 use $t_{end} = 8$ s → $f_{min} = 0.125$ Hz; test-4 uses $t_{end} = 12$ s → $f_{min} = 0.083$ Hz
- $f_{max} = c_{s,min} / (\text{PPW} \times \Delta x)$
- $f_{Nyq} = c_{s,min} / (2\,\Delta x)$
- For elastic tests (tests 1–3), $c_{s} = 3.464$ km/s raises all $f_{max}$ and $f_{Nyq}$ limits by ~15%
- Content between $f_{max}$ practical and $f_{max}$ upper is present in the simulation but subject to numerical dispersion
