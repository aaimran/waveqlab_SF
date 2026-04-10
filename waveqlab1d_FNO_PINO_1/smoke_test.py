import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("1. importing torch...")
import torch
print("   torch", torch.__version__)

print("2. importing model...")
from model import SeparateFNO2d, UnifiedFNO2d, PINOLoss
print("   model OK")

print("3. forward pass SeparateFNO2d...")
m = SeparateFNO2d(c_in=15, modes_x=16, modes_t=24, width=64, n_layers=4, c_out=4)
x = torch.randn(2, 15, 32, 32)
y = m(x)
print("   output shape:", y.shape)

print("4. forward pass UnifiedFNO2d...")
um = UnifiedFNO2d(c_in=22, modes_x=24, modes_t=32, width=64, n_layers=4, c_out=4,
                  n_fric=2, n_bc=3, embed_dim=16)
fi = torch.zeros(2, dtype=torch.long)
bi = torch.ones(2, dtype=torch.long)
uy = um(torch.randn(2, 22, 32, 32), fi, bi)
print("   output shape:", uy.shape)

print("5. importing data_gen...")
from data_gen.param_space import sample_sw, sample_rs, sample_unified
sw = sample_sw(4, seed=0)
rs = sample_rs(4, seed=0)
print("   SW sample keys:", list(sw[0].keys()))
print("   RS sample keys:", list(rs[0].keys()))

print("\nAll smoke tests PASSED.")
