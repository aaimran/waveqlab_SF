"""
fno.py — FNO-2D architecture for rupture field prediction.

Architecture:
    Input  : (batch, C_in, NX, NT)  where C_in = N_params + 2  (params + x,t coords)
    Output : (batch, 4, NX, NT)     — (v_l, s_l, v_r, s_r)

Components:
    SpectralConv2d  — Fourier integral operator (FFT2 → complex weight → IFFT2)
    FNOBlock2d      — SpectralConv2d + local linear bypass + activation
    FNO2d           — lifting → n_layers × FNOBlock2d → projection

Reference: Li et al. (2021) "Fourier Neural Operator for Parametric PDEs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier integral operator layer.

    Computes:
        (R_phi * x)(x,t) = IFFT2 [ R_phi(k_x, k_t) * FFT2[x](k_x, k_t) ]

    Only the first `modes_x` × `modes_t` Fourier modes are retained (low-pass truncation).
    Uses rfft2/irfft2 so weights are stored for the upper half of the spectrum only.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_t: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t

        # Complex weights for the four quadrants of the rfft2 output:
        #   (k_x in [0, modes_x),   k_t in [0,       modes_t))
        #   (k_x in [-modes_x, 0),  k_t in [0,       modes_t))
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_t, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_t, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(batch, in_ch, kx, kt) × (in_ch, out_ch, kx, kt) → (batch, out_ch, kx, kt)."""
        return torch.einsum('bixt,ioxt->boxt', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_channels, NX, NT) → (batch, out_channels, NX, NT)."""
        batch, _, nx, nt = x.shape

        # FFT2 over spatial (dim=-2) and temporal (dim=-1) axes
        x_ft = torch.fft.rfft2(x, norm='ortho')   # (B, C, NX, NT//2+1)

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            batch, self.out_channels, nx, nt // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Low-frequency block (positive k_x, positive k_t)
        out_ft[:, :, :self.modes_x, :self.modes_t] = self._compl_mul2d(
            x_ft[:, :, :self.modes_x, :self.modes_t], self.weights1
        )
        # High-frequency block (negative k_x, positive k_t) — wrap around
        out_ft[:, :, -self.modes_x:, :self.modes_t] = self._compl_mul2d(
            x_ft[:, :, -self.modes_x:, :self.modes_t], self.weights2
        )

        # IFFT2 back to physical space
        return torch.fft.irfft2(out_ft, s=(nx, nt), norm='ortho')   # (B, C, NX, NT)


class FNOBlock2d(nn.Module):
    """One FNO layer: SpectralConv2d + point-wise linear bypass + GeLU."""

    def __init__(self, width: int, modes_x: int, modes_t: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_t)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.bypass(x))


class FNO2d(nn.Module):
    """
    FNO-2D for rupture field prediction.

    Input  : (batch, C_in,  NX, NT)  — parameter channels + coordinate channels
    Output : (batch, C_out, NX, NT)  — (v_l, s_l, v_r, s_r)

    Parameters
    ----------
    n_params  : int   number of physical input parameters (e.g. 4)
    modes_x   : int   Fourier modes kept in space
    modes_t   : int   Fourier modes kept in time
    width     : int   hidden channel dimensionality (d_v)
    n_layers  : int   number of FNO blocks
    c_out     : int   output channels (default 4)
    """

    def __init__(
        self,
        n_params : int  = 4,
        modes_x  : int  = 16,
        modes_t  : int  = 16,
        width    : int  = 64,
        n_layers : int  = 4,
        c_out    : int  = 4,
    ):
        super().__init__()
        self.n_params = n_params
        self.c_in     = n_params + 2   # param channels + (x, t)
        self.width    = width

        # Lifting layer
        self.lift = nn.Conv2d(self.c_in, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes_x, modes_t) for _ in range(n_layers)]
        )

        # Projection layers (two-layer MLP per pixel)
        self.proj1 = nn.Conv2d(width, width * 2, kernel_size=1)
        self.proj2 = nn.Conv2d(width * 2, c_out, kernel_size=1)

    def forward(
        self,
        params: torch.Tensor,    # (batch, N_params)
        coords: torch.Tensor,    # (batch, 2, NX, NT)  — normalised (x,t) in [0,1]
    ) -> torch.Tensor:
        """Returns (batch, 4, NX, NT) — predicted (v_l, s_l, v_r, s_r)."""
        batch, _, nx, nt = coords.shape

        # Broadcast parameter vector over spatial-temporal grid
        # (batch, N_params) → (batch, N_params, NX, NT)
        p_grid = params.view(batch, self.n_params, 1, 1).expand(batch, self.n_params, nx, nt)

        # Concatenate with coordinate channels
        x = torch.cat([p_grid, coords], dim=1)   # (batch, C_in, NX, NT)

        # Lifting
        x = self.lift(x)                          # (batch, width, NX, NT)

        # FNO blocks
        for block in self.blocks:
            x = block(x)                          # (batch, width, NX, NT)

        # Projection MLP
        x = F.gelu(self.proj1(x))                 # (batch, width*2, NX, NT)
        x = self.proj2(x)                         # (batch, c_out, NX, NT)

        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = FNO2d(n_params=4, modes_x=16, modes_t=16, width=64, n_layers=4)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'FNO2d parameters: {n_total:,}')

    params = torch.randn(2, 4)
    coords = torch.rand(2, 2, 64, 64)
    out = model(params, coords)
    print(f'Output shape: {out.shape}')  # expected: (2, 4, 64, 64)
