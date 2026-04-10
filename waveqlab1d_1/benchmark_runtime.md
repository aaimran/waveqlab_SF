# waveqlab1d Runtime Benchmark

**Platform:** serial (1 CPU), `NUMBA_NUM_THREADS=1`  
**Solver:** waveqlab1d (Numba, order-6 SBP-SAT)  

| Test    | Resolution | BC                     | Total Grid Points | Total Time Steps | Total Work | Run Time (excl. JIT warmup) | JIT Warm-up Time | Avg. Run Time per step |
| :------ | :--------: | :--------------------- | ----------------: | ---------------: | ---------: | --------------------------: | ---------------: | ---------------------: |
| test-1a |    80m     | Free surface           |               752 |              693 |    521,136 |                      0.07 s |           0.27 s |               0.097 ms |
| test-1a |    40m     | Free surface           |             1,502 |            1,386 |  2,081,772 |                      0.27 s |           0.27 s |               0.182 ms |
| test-1a |    20m     | Free surface           |             3,002 |            2,771 |  8,318,542 |                      1.06 s |           0.27 s |               0.353 ms |
| test-1b |    80m     | Absorbing              |               752 |              693 |    521,136 |                      0.07 s |           0.27 s |               0.097 ms |
| test-1b |    40m     | Absorbing              |             1,502 |            1,386 |  2,081,772 |                      0.27 s |           0.27 s |               0.182 ms |
| test-1b |    20m     | Absorbing              |             3,002 |            2,771 |  8,318,542 |                      1.05 s |           0.27 s |               0.352 ms |
| test-1c |    80m     | PML 5 km               |               752 |              693 |    521,136 |                      0.07 s |           0.27 s |               0.099 ms |
| test-1c |    40m     | PML 5 km               |             1,502 |            1,386 |  2,081,772 |                      0.27 s |           0.27 s |               0.182 ms |
| test-1c |    20m     | PML 5 km               |             3,002 |            2,771 |  8,318,542 |                      1.06 s |           0.27 s |               0.353 ms |
| test-2a |    80m     | Free surface           |               752 |              693 |    521,136 |                      0.10 s |           0.27 s |               0.141 ms |
| test-2a |    40m     | Free surface           |             1,502 |            1,386 |  2,081,772 |                      0.39 s |           0.26 s |               0.265 ms |
| test-2a |    20m     | Free surface           |             3,002 |            2,771 |  8,318,542 |                      1.58 s |           0.26 s |               0.525 ms |
| test-2b |    80m     | Absorbing              |               752 |              693 |    521,136 |                      0.10 s |           0.26 s |               0.141 ms |
| test-2b |    40m     | Absorbing              |             1,502 |            1,386 |  2,081,772 |                      0.39 s |           0.26 s |               0.265 ms |
| test-2b |    20m     | Absorbing              |             3,002 |            2,771 |  8,318,542 |                      1.59 s |           0.26 s |               0.526 ms |
| test-2c |    80m     | PML 5 km               |               752 |              693 |    521,136 |                      0.10 s |           0.26 s |               0.141 ms |
| test-2c |    40m     | PML 5 km               |             1,502 |            1,386 |  2,081,772 |                      0.38 s |           0.26 s |               0.264 ms |
| test-2c |    20m     | PML 5 km               |             3,002 |            2,771 |  8,318,542 |                      1.60 s |           0.27 s |               0.530 ms |
| test-3a |    80m     | Free surface           |               752 |              693 |    521,136 |                      0.11 s |           0.27 s |               0.145 ms |
| test-3a |    40m     | Free surface           |             1,502 |            1,386 |  2,081,772 |                      0.38 s |           0.26 s |               0.263 ms |
| test-3a |    20m     | Free surface           |             3,002 |            2,771 |  8,318,542 |                      1.59 s |           0.27 s |               0.531 ms |
| test-3b |    80m     | Absorbing              |               752 |              693 |    521,136 |                      0.10 s |           0.27 s |               0.141 ms |
| test-3b |    40m     | Absorbing              |             1,502 |            1,386 |  2,081,772 |                      0.38 s |           0.26 s |               0.263 ms |
| test-3b |    20m     | Absorbing              |             3,002 |            2,771 |  8,318,542 |                      1.59 s |           0.27 s |               0.528 ms |
| test-3c |    80m     | PML 5 km               |               752 |              693 |    521,136 |                      0.10 s |           0.26 s |               0.142 ms |
| test-3c |    40m     | PML 5 km               |             1,502 |            1,386 |  2,081,772 |                      0.38 s |           0.26 s |               0.264 ms |
| test-3c |    20m     | PML 5 km               |             3,002 |            2,771 |  8,318,542 |                      1.60 s |           0.26 s |               0.531 ms |
| test-4a |    80m     | Free surface           |               752 |            1,039 |    781,328 |                      0.15 s |           0.26 s |               0.141 ms |
| test-4a |    40m     | Free surface           |             1,502 |            2,078 |  3,121,156 |                      0.58 s |           0.26 s |               0.262 ms |
| test-4a |    20m     | Free surface           |             3,002 |            4,157 | 12,479,314 |                      2.34 s |           0.26 s |               0.518 ms |
| test-4b |    80m     | Absorbing              |               752 |            1,039 |    781,328 |                      0.16 s |           0.26 s |               0.143 ms |
| test-4b |    40m     | Absorbing              |             1,502 |            2,078 |  3,121,156 |                      0.57 s |           0.26 s |               0.261 ms |
| test-4b |    20m     | Absorbing              |             3,002 |            4,157 | 12,479,314 |                      2.35 s |           0.27 s |               0.519 ms |
| test-4c |    80m     | PML 5 km               |               752 |            1,039 |    781,328 |                      0.16 s |           0.26 s |               0.142 ms |
| test-4c |    40m     | PML 5 km               |             1,502 |            2,078 |  3,121,156 |                      0.57 s |           0.26 s |               0.262 ms |
| test-4c |    20m     | PML 5 km               |             3,002 |            4,157 | 12,479,314 |                      2.34 s |           0.26 s |               0.518 ms |
| test-1r |    80m     | Free surface (ref 2×L) |             1,502 |              693 |  1,040,886 |                      0.14 s |           0.27 s |               0.185 ms |
| test-1r |    40m     | Free surface (ref 2×L) |             3,002 |            1,386 |  4,160,772 |                      0.53 s |           0.26 s |               0.362 ms |
| test-1r |    20m     | Free surface (ref 2×L) |             6,002 |            2,771 | 16,631,542 |                      2.20 s |           0.28 s |               0.710 ms |
| test-2r |    80m     | Free surface (ref 2×L) |             1,502 |              693 |  1,040,886 |                      0.19 s |           0.26 s |               0.269 ms |
| test-2r |    40m     | Free surface (ref 2×L) |             3,002 |            1,386 |  4,160,772 |                      0.83 s |           0.27 s |               0.563 ms |
| test-2r |    20m     | Free surface (ref 2×L) |             6,002 |            2,771 | 16,631,542 |                      3.32 s |           0.26 s |               1.103 ms |
| test-3r |    80m     | Free surface (ref 2×L) |             1,502 |              693 |  1,040,886 |                      0.19 s |           0.27 s |               0.271 ms |
| test-3r |    40m     | Free surface (ref 2×L) |             3,002 |            1,386 |  4,160,772 |                      0.82 s |           0.27 s |               0.559 ms |
| test-3r |    20m     | Free surface (ref 2×L) |             6,002 |            2,771 | 16,631,542 |                      3.33 s |           0.27 s |               1.108 ms |
| test-4r |    80m     | Free surface (ref 2×L) |             1,502 |            1,039 |  1,560,578 |                      0.29 s |           0.26 s |               0.266 ms |
| test-4r |    40m     | Free surface (ref 2×L) |             3,002 |            2,078 |  6,238,156 |                      1.22 s |           0.26 s |               0.547 ms |
| test-4r |    20m     | Free surface (ref 2×L) |             6,002 |            4,157 | 24,950,314 |                      4.81 s |           0.26 s |               1.069 ms |
