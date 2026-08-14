import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class BidirectionalMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.act = nn.SiLU()

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        self.conv1d_a = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_a = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.dt_proj_a = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        self.dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_a.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_a.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_a.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_a.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True

        # S4D real initialization
        A_a = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_a_log = torch.log(A_a)  # Keep A_log in fp32
        self.A_a_log = nn.Parameter(A_a_log)
        self.A_a_log._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        # D "skip" parameter
        self.D_a = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_a._no_weight_decay = True

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A_a = -torch.exp(self.A_a_log.float())  # (d_inner, d_state)
        A_b = -torch.exp(self.A_b_log.float())

        # a

        x_a, z_a = xz.chunk(2, dim=1)

        x_a = self.conv1d_a(x_a)[..., :seqlen]
        x_a = self.act(x_a)

        x_a_dbl = self.x_proj_a(rearrange(x_a, "b d l -> (b l) d"))  # (bl d)
        dt_a, B_a, C_a = torch.split(
            x_a_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_a = self.dt_proj_a.weight @ dt_a.t()
        dt_a = rearrange(dt_a, "d (b l) -> b d l", l=seqlen)
        B_a = rearrange(B_a, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_a = rearrange(C_a, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y_a = selective_scan_fn(
            x_a,
            dt_a,
            A_a,
            B_a,
            C_a,
            self.D_a.float(),
            delta_bias=self.dt_proj_a.bias.float(),
            delta_softplus=True,
        )

        y_a = y_a * self.act(z_a)

        # b

        xz_b = xz.flip([-1])
        x_b, z_b = xz_b.chunk(2, dim=1)

        x_b = self.conv1d_b(x_b)[..., :seqlen]
        x_b = self.act(x_b)

        x_b_dbl = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
        dt_b, B_b, C_b = torch.split(
            x_b_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_b = self.dt_proj_b.weight @ dt_b.t()
        dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
        B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y_b = selective_scan_fn(
            x_b,
            dt_b,
            A_b,
            B_b,
            C_b,
            self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )

        y_b = y_b * self.act(z_b)

        y_b = y_b.flip([-1])

        y = y_a + y_b
        y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)

        return out
