"""
planC/model/fno.py — FNO-2D with Plan-C upgrades:
  - Mode clamping (modes capped at available FFT frequencies at any grid size)
  - Spatial profile channels broadcast over time (same design as Plan C)
  - n_spatial_channels parameter to accept per-grid-point material fields
  - Otherwise identical structure to the stable 1d_numba_PINO_FNO_0 FNO2d

Input layout  (batch, C_in, NX, NT)
  C_in = n_spatial + n_scalar + 2  (spatial profiles + scalar params + x,t coords)
Output layout (batch, 4, NX, NT)  → (v_l, s_l, v_r, s_r)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier integral operator.  Modes are CLAMPED to available FFT
    frequencies at runtime so the same weights work at any (NX, NT).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_t: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t

        scale = 1.0 / (in_channels * out_channels)
        self.W1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_t,
                               dtype=torch.cfloat))
        self.W2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_t,
                               dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bixt,ioxt->boxt', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, nt = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')   # (B, C, NX, NT//2+1)

        # Clamp modes to available FFT frequencies — enables super-resolution
        mx = min(self.modes_x, nx // 2)
        mt = min(self.modes_t, nt // 2 + 1)

        out_ft = torch.zeros(batch, self.out_channels, nx, nt // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :mt]  = self._mul(x_ft[:, :, :mx, :mt],
                                             self.W1[:, :, :mx, :mt])
        out_ft[:, :, -mx:, :mt] = self._mul(x_ft[:, :, -mx:, :mt],
                                             self.W2[:, :, :mx, :mt])

        return torch.fft.irfft2(out_ft, s=(nx, nt), norm='ortho')


class FNOBlock2d(nn.Module):
    """SpectralConv2d + 1×1 bypass + GeLU."""

    def __init__(self, width: int, modes_x: int, modes_t: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_t)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.bypass(x))


class FNO2d(nn.Module):
    """
    FNO-2D for Plan-C rupture prediction.

    Accepts a pre-built input tensor of shape (B, C_in, NX, NT) where C_in
    already includes spatial channels, scalar channels, and (x, t) coords.
    This makes it resolution-agnostic: just change the grid, modes are clamped.

    Parameters
    ----------
    c_in     : total input channels
    modes_x  : Fourier modes in space (soft max; clamped at runtime)
    modes_t  : Fourier modes in time
    width    : hidden channel width
    n_layers : FNO block repetitions
    c_out    : output channels (default 4: v_l, s_l, v_r, s_r)
    """

    def __init__(self, c_in: int = 9, modes_x: int = 24, modes_t: int = 24,
                 width: int = 64, n_layers: int = 4, c_out: int = 4):
        super().__init__()
        self.c_in  = c_in
        self.width = width

        self.lift   = nn.Conv2d(c_in, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes_x, modes_t) for _ in range(n_layers)])
        self.proj1  = nn.Conv2d(width, width * 2, kernel_size=1)
        self.proj2  = nn.Conv2d(width * 2, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C_in, NX, NT) → (B, 4, NX, NT)"""
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj2(F.gelu(self.proj1(x)))


# ── Input tensor builder ─────────────────────────────────────────────────────

def build_input_tensor(
    params_raw: torch.Tensor,   # (B, N_scalar)
    coords: torch.Tensor,       # (B, 2, NX, NT)  — normalised x, t
    rho_field: torch.Tensor     = None,  # (B, NX) optional
    mu_field: torch.Tensor      = None,  # (B, NX) optional
) -> torch.Tensor:
    """
    Build (B, C_in, NX, NT) input tensor from scalar params + coords.
    Optionally includes rho(x) and mu(x) as spatial channels broadcast over t.
    """
    B, _, nx, nt = coords.shape

    # Broadcast scalar params over full (NX, NT) grid
    ch = [params_raw[:, i].view(B, 1, 1, 1).expand(B, 1, nx, nt)
          for i in range(params_raw.shape[1])]

    if rho_field is not None:
        ch.append(rho_field.view(B, 1, nx, 1).expand(B, 1, nx, nt))
    if mu_field is not None:
        ch.append(mu_field.view(B, 1, nx, 1).expand(B, 1, nx, nt))

    ch.append(coords)     # (B, 2, NX, NT)
    return torch.cat(ch, dim=1)


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    c_in = 7   # 5 SW scalars + 2 coords
    m = FNO2d(c_in=c_in, modes_x=24, modes_t=24, width=64, n_layers=4)
    n = sum(p.numel() for p in m.parameters())
    print(f'FNO2d(c_in={c_in}) params: {n:,}')
    x = torch.randn(2, c_in, 80, 80)
    o = m(x)
    print(f'Output: {o.shape}')   # (2, 4, 80, 80)
    # Resolution invariance test
    x2 = torch.randn(2, c_in, 160, 160)
    o2 = m(x2)
    print(f'Output HR: {o2.shape}')  # (2, 4, 160, 160)
