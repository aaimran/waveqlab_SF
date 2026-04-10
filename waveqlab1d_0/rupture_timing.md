# Rupture Timing and Boundary Interaction

## Fault Setup

- Fault located at center: $x = L/2 = 15$ km
- Friction law: Slip-Weakening (SW)
- $\tau_0 = 81.6$ MPa, $\sigma_n = 120$ MPa, $\alpha_s = 0.677$ → $\tau_s = 81.24$ MPa
- Slip begins immediately at $t = 0$ (fault is loaded above static yield stress)
- Rupture expands **bilaterally** from the fault center

## Rupture Velocity

SW friction produces sub-shear rupture, typically $v_r \approx 0.85\, c_s$:

$$v_r \approx 0.85 \times 3.464 \text{ km/s} \approx 2.94 \text{ km/s}$$

## Time to Reach Boundary

Distance from fault center to each boundary = 15 km:

$$t_{reach} = \frac{15 \text{ km}}{0.85 \times 3.464 \text{ km/s}} \approx 5.1 \text{ s}$$

**The rupture front reaches both boundaries at ~5 s**, well before $t_{end} = 8$ s.

## Significance of the Final ~3 Seconds

The window from $t \approx 5$ s to $t_{end} = 8$ s captures:

- Reflected waves returning from the boundaries
- Post-rupture healing and arrest dynamics
- **This is the key reason tests a/b/c differ** — the three boundary conditions produce different reflected wavefields during this interval:
  - **a (Free surface):** reflects with amplitude doubling — strong contamination
  - **b (Absorbing):** damps outgoing waves — minimal reflection
  - **c (PML 5 km):** attenuates outgoing waves — near-zero reflection

## Test-4 Timing

Test-4 uses $t_{end} = 12$ s and $c_{s,min} = 3.0$ km/s (slow layer):

$$t_{reach} = \frac{15 \text{ km}}{0.85 \times 3.0 \text{ km/s}} \approx 5.9 \text{ s}$$

This gives **6+ seconds of post-rupture wave propagation**, making boundary effects even more pronounced and validation of attenuation more reliable.
