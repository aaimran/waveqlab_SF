"""
fno.py — FNO-2D backbone for waveqlab1d PINO (Plans A and B)
=============================================================

Two model classes:
  UnifiedFNO2d  — Plan A: handles all friction laws and BCs via FiLM conditioning
  SeparateFNO2d — Plan B: fixed friction law + BC, smaller and faster

Architecture:
    Input  : (batch, C_in, NX, NT)
    Output : (batch, 4,    NX, NT)  — (v_l, s_l, v_r, s_r)

Common components:
    SpectralConv2d — 2D Fourier integral operator (rfft2 / irfft2)
    FNOBlock2d     — SpectralConv2d + 1×1 bypass + optional FiLM + GeLU
    FiLMLayer      — Feature-wise Linear Modulation: γ·x + β from embedding

Reference: Li et al. (2021) "Fourier Neural Operator for Parametric PDEs"
FiLM:      Perez et al. (2018) "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Spectral convolution ─────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    2D Fourier integral operator.

    Retains only the first modes_x × modes_t Fourier modes (low-pass truncation).
    Uses rfft2/irfft2 to halve memory for the real-valued input.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_t: int):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.modes_x = modes_x
        self.modes_t = modes_t

        scale = 1.0 / (in_ch * out_ch)
        # Two weight blocks: positive and negative k_x quadrants
        self.W1 = nn.Parameter(scale * torch.rand(
            in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat))
        self.W2 = nn.Parameter(scale * torch.rand(
            in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(B, C_in, kx, kt) × (C_in, C_out, kx, kt) → (B, C_out, kx, kt)."""
        return torch.einsum('bixt,ioxt->boxt', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C_in, NX, NT) → (B, C_out, NX, NT)."""
        B, _, nx, nt = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')   # (B, C, NX, NT//2+1)

        # Clamp modes to what is available for this resolution
        mx = min(self.modes_x, nx // 2)
        mt = min(self.modes_t, nt // 2 + 1)

        out_ft = torch.zeros(B, self.out_ch, nx, nt // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :mt] = self._mul(
            x_ft[:, :, :mx, :mt], self.W1[:, :, :mx, :mt])
        out_ft[:, :, -mx:, :mt] = self._mul(
            x_ft[:, :, -mx:, :mt], self.W2[:, :, :mx, :mt])

        return torch.fft.irfft2(out_ft, s=(nx, nt), norm='ortho')


# ─── FiLM conditioning ────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: y = γ(e) ⊙ x + β(e)

    Maps a conditioning embedding e → per-channel scale (γ) and shift (β).
    Broadcast over the spatial/temporal dimensions of the feature map.
    """

    def __init__(self, embed_dim: int, width: int):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, width)
        self.beta  = nn.Linear(embed_dim, width)
        # Initialise scale near 1, shift near 0
        nn.init.ones_(self.gamma.weight);  nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        x : (B, W, NX, NT)
        e : (B, embed_dim)
        returns : (B, W, NX, NT)
        """
        γ = self.gamma(e)[:, :, None, None]   # (B, W, 1, 1)
        β = self.beta(e)[:, :, None, None]
        return γ * x + β


# ─── FNO blocks ───────────────────────────────────────────────────────────────

class FNOBlock2d(nn.Module):
    """
    One FNO layer: SpectralConv2d + 1×1 bypass + optional FiLM + GeLU.

    If film is not None, applies FiLM modulation *before* GeLU.
    """

    def __init__(self, width: int, modes_x: int, modes_t: int,
                 film: FiLMLayer | None = None):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_t)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.film     = film

    def forward(self, x: torch.Tensor,
                e: torch.Tensor | None = None) -> torch.Tensor:
        h = self.spectral(x) + self.bypass(x)
        if self.film is not None and e is not None:
            h = self.film(h, e)
        return F.gelu(h)


# ─── Plan B: SeparateFNO2d ────────────────────────────────────────────────────

class SeparateFNO2d(nn.Module):
    """
    FNO-2D for a fixed friction law and BC type (Plan B).
    No FiLM conditioning — the model is specialized to one (fric_law, bc) combo.

    Parameters
    ----------
    c_in     : int   input channels (see plan_B_separate.md §3)
    modes_x  : int   spectral modes in space
    modes_t  : int   spectral modes in time
    width    : int   hidden dimension
    n_layers : int   number of FNO blocks
    c_out    : int   output channels (default 4)
    """

    def __init__(
        self,
        c_in    : int  = 15,
        modes_x : int  = 16,
        modes_t : int  = 24,
        width   : int  = 64,
        n_layers: int  = 4,
        c_out   : int  = 4,
    ):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes_x, modes_t) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, c_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C_in, NX, NT) → (B, 4, NX, NT)."""
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Plan A: UnifiedFNO2d ─────────────────────────────────────────────────────

class UnifiedFNO2d(nn.Module):
    """
    FNO-2D with FiLM conditioning for all friction laws and BCs (Plan A).

    Discrete conditions are encoded as learned embeddings and modulate each
    FNO block via FiLM (scale + shift per channel).

    Parameters
    ----------
    c_in      : int   continuous input channels (spatial + scalar + coords = 22)
    modes_x   : int   spectral modes in space
    modes_t   : int   spectral modes in time
    width     : int   hidden dimension
    n_layers  : int   number of FNO blocks
    c_out     : int   output channels (default 4)
    n_fric    : int   number of friction law types (2: SW, RS)
    n_bc      : int   number of outer BC types (3: free, absorb, PML)
    embed_dim : int   FiLM embedding dimension
    """

    # Discrete index maps
    FRIC_IDX = {'SW': 0, 'RS': 1}
    BC_IDX   = {'free': 0, 'absorbing': 1, 'pml': 2}

    def __init__(
        self,
        c_in     : int  = 22,
        modes_x  : int  = 24,
        modes_t  : int  = 32,
        width    : int  = 128,
        n_layers : int  = 6,
        c_out    : int  = 4,
        n_fric   : int  = 2,
        n_bc     : int  = 3,
        embed_dim: int  = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Discrete embeddings
        self.fric_embed = nn.Embedding(n_fric, embed_dim)
        self.bc_embed   = nn.Embedding(n_bc,   embed_dim)
        # Combined FiLM input: 2 × embed_dim
        film_in = 2 * embed_dim

        # Build FiLM layers (one per block)
        films = [FiLMLayer(film_in, width) for _ in range(n_layers)]

        self.lift = nn.Conv2d(c_in, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes_x, modes_t, film=films[i])
             for i in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, c_out, kernel_size=1),
        )

    def encode_condition(
        self,
        fric_idx: torch.Tensor,   # (B,) int64
        bc_idx  : torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        """Returns combined embedding (B, 2*embed_dim)."""
        e_fric = self.fric_embed(fric_idx)   # (B, embed_dim)
        e_bc   = self.bc_embed(bc_idx)       # (B, embed_dim)
        return torch.cat([e_fric, e_bc], dim=-1)   # (B, 2*embed_dim)

    def forward(
        self,
        x       : torch.Tensor,   # (B, C_in, NX, NT)
        fric_idx: torch.Tensor,   # (B,) int64  — friction law index
        bc_idx  : torch.Tensor,   # (B,) int64  — BC mode index
    ) -> torch.Tensor:
        """→ (B, 4, NX, NT)."""
        e = self.encode_condition(fric_idx, bc_idx)   # (B, 2*embed_dim)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h, e)
        return self.proj(h)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def fric_tensor(fric_law: str, batch: int, device) -> torch.Tensor:
        """Utility: make (batch,) int64 tensor for a friction law string."""
        idx = UnifiedFNO2d.FRIC_IDX[fric_law.upper()]
        return torch.full((batch,), idx, dtype=torch.long, device=device)

    @staticmethod
    def bc_tensor(bc_mode: str, batch: int, device) -> torch.Tensor:
        """Utility: make (batch,) int64 tensor for a BC mode string."""
        idx = UnifiedFNO2d.BC_IDX[bc_mode.lower()]
        return torch.full((batch,), idx, dtype=torch.long, device=device)


# ─── Super-resolution utilities ───────────────────────────────────────────────

def interpolate_to_resolution(
    u: torch.Tensor,
    nx_out: int,
    nt_out: int,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Upsample or downsample a field tensor to a target resolution.

    u       : (B, C, NX, NT)
    returns : (B, C, nx_out, nt_out)
    """
    return F.interpolate(u, size=(nx_out, nt_out),
                         mode=mode, align_corners=False)


def build_coord_channels(nx: int, nt: int, device) -> torch.Tensor:
    """
    Returns (2, NX, NT) tensor with x=[0,1] and t=[0,1] coordinate maps.
    Used as the last two input channels for the FNO.
    """
    x = torch.linspace(0, 1, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    x_map = x[:, None].expand(nx, nt)   # (NX, NT)
    t_map = t[None, :].expand(nx, nt)   # (NX, NT)
    return torch.stack([x_map, t_map], dim=0)   # (2, NX, NT)


def build_input_tensor_unified(
    cs_arr     : torch.Tensor,    # (NX,)
    rho_arr    : torch.Tensor,    # (NX,)
    mu_arr     : torch.Tensor,    # (NX,)
    Qs_inv_arr : torch.Tensor,    # (NX,)
    d_l        : torch.Tensor,    # (NX,)
    d_r        : torch.Tensor,    # (NX,)
    scalars    : torch.Tensor,    # (14,)  see plan_A_unified.md §3
    nx         : int,
    nt         : int,
) -> torch.Tensor:
    """
    Assemble (1, C_in=22, NX, NT) input tensor for UnifiedFNO2d.

    Spatial profiles are broadcast over time; scalars over both dims.
    """
    device = cs_arr.device

    # Spatial profiles: (NX,) → (NX, NT)
    def bcast_x(v):
        return v[:, None].expand(nx, nt)

    spatial = torch.stack([
        bcast_x(cs_arr), bcast_x(rho_arr), bcast_x(mu_arr),
        bcast_x(Qs_inv_arr), bcast_x(d_l), bcast_x(d_r),
    ], dim=0)   # (6, NX, NT)

    # Scalar params: (14,) → (14, NX, NT)
    scalar_maps = scalars[:, None, None].expand(14, nx, nt)   # (14, NX, NT)

    # Coordinates
    coords = build_coord_channels(nx, nt, device)   # (2, NX, NT)

    combined = torch.cat([spatial, scalar_maps, coords], dim=0)   # (22, NX, NT)
    return combined.unsqueeze(0)   # (1, 22, NX, NT)


def build_input_tensor_separate(
    cs_arr     : torch.Tensor,    # (NX,)
    rho_arr    : torch.Tensor,    # (NX,)
    mu_arr     : torch.Tensor,    # (NX,)
    Qs_inv_arr : torch.Tensor,    # (NX,)
    d_l        : torch.Tensor,    # (NX,)
    d_r        : torch.Tensor,    # (NX,)
    scalars    : torch.Tensor,    # (n_scalar,) — depends on fric_law
    nx         : int,
    nt         : int,
) -> torch.Tensor:
    """
    Assemble (1, C_in, NX, NT) input tensor for SeparateFNO2d.
    C_in = 6 (spatial) + len(scalars) + 2 (coords).
    """
    device = cs_arr.device

    def bcast_x(v):
        return v[:, None].expand(nx, nt)

    spatial = torch.stack([
        bcast_x(cs_arr), bcast_x(rho_arr), bcast_x(mu_arr),
        bcast_x(Qs_inv_arr), bcast_x(d_l), bcast_x(d_r),
    ], dim=0)   # (6, NX, NT)

    n_sc = scalars.shape[0]
    scalar_maps = scalars[:, None, None].expand(n_sc, nx, nt)   # (n_sc, NX, NT)
    coords = build_coord_channels(nx, nt, device)               # (2, NX, NT)

    combined = torch.cat([spatial, scalar_maps, coords], dim=0)
    return combined.unsqueeze(0)
