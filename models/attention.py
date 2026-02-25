import math

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

from xai.lrp_rules import modified_linear_layer
from xai.lrp_utils import set_lrp_params


def exists(val):
    """
    (c) copied from https://github.com/lucidrains/nystrom-attention
    """
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    """
    (c) copied from https://github.com/lucidrains/nystrom-attention
    """

    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class Attention(nn.Module):
    """
    (c) please also check https://github.com/lucidrains/nystrom-attention
    """

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
        method="nystrom",
        bias=True,
    ):
        if method not in ["nystrom", "dot_prod"]:
            raise ValueError(
                "Only Nystrom and dot product attention can be used. "
                "Set attention method to 'nysterom' or 'dot_prod'"
            )
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (kernel_size, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

        self.attn_scores = None
        self.padding = None
        self.method = method

    def self_attention(
        self,
        x,
        mask=None,
        xai_mode=False,
        lrp_params=None,
        save_attn=False,
        verbose=False,
    ):

        lrp_params = set_lrp_params(lrp_params)

        b, n, _ = x.shape
        h, m, iters, eps = (
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )

        if self.method == "nystrom":
            # pad so that sequence can be evenly divided into m landmarks
            remainder = n % m
            if remainder > 0:
                padding = m - (n % m)
                x = F.pad(x, (0, 0, padding, 0), value=0)
                if not xai_mode and save_attn:
                    self.padding = padding

                if exists(mask):
                    mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        if xai_mode:
            to_qkv_ = modified_linear_layer(
                self.to_qkv, lrp_params["gamma"], lrp_params["no_bias"]
            )
            q, k, v = to_qkv_(x).chunk(3, dim=-1)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.method == "nystrom":
            # set masked positions to 0 in queries, keys, values
            if exists(mask):
                mask = rearrange(mask, "b n -> b () n")
                q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        if self.method == "nystrom":
            # generate landmarks by sum reduction, and then calculate mean using the mask
            l = math.ceil(n / m)
            landmark_einops_eq = "... (n l) d -> ... n d"
            q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
            k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

            # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

            divisor = l
            if exists(mask):
                mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
                divisor = mask_landmarks_sum[..., None] + eps
                mask_landmarks = mask_landmarks_sum > 0

            # masked mean (if mask exists)

            q_landmarks /= divisor
            k_landmarks /= divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        if self.method == "nystrom":
            sim1 = einsum(einops_eq, q, k_landmarks)
            sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
            sim3 = einsum(einops_eq, q_landmarks, k)

            # masking
            if exists(mask):
                mask_value = -torch.finfo(q.dtype).max
                sim1.masked_fill_(
                    ~(mask[..., None] * mask_landmarks[..., None, :]), mask_value
                )
                sim2.masked_fill_(
                    ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]),
                    mask_value,
                )
                sim3.masked_fill_(
                    ~(mask_landmarks[..., None] * mask[..., None, :]), mask_value
                )

        else:
            sim1 = einsum(einops_eq, q, k)

        if xai_mode and save_attn:
            # assumption: in xai_mode we anyway don't want to save attention scores
            save_attn = False
            if verbose:
                print(
                    "in xai mode no attention score can be saved! therefore, save_attn is set to False"
                )

        # eq (15) in the paper and aggregate values
        if self.method == "nystrom":
            attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
            attn2_inv = moore_penrose_iter_pinv(attn2, iters)

            if not xai_mode and not save_attn:
                # this if case shall be used for the usual forward pass in training
                out = (attn1 @ attn2_inv) @ (attn3 @ v)
            elif not xai_mode and save_attn:
                # this if case shall be used for computing the attention map
                self.attn_scores = attn1 @ attn2_inv @ attn3
                out = self.attn_scores @ v
            else:
                # this if case shall be used in xai mode
                out = (attn1 @ attn2_inv @ attn3).detach() @ v
        else:  # dot-prod attention
            if not xai_mode and not save_attn:
                # this if case shall be used for the usual forward pass in training
                out = F.softmax(sim1, dim=-1) @ v
            elif not xai_mode and save_attn:
                # this if case shall be used for computing the attention map
                self.attn_scores = F.softmax(sim1, dim=-1)
                out = self.attn_scores @ v
            else:
                # this if case shall be used in xai mode
                out = F.softmax(sim1, dim=-1).detach() @ v

        if xai_mode and verbose:
            print("Attention heads were detached from the computational graph!")

        return out, v

    def forward(self, x, mask=None, save_attn=False):

        out, v = self.self_attention(x, mask=mask, save_attn=save_attn)

        # add depth-wise conv residual of values
        if self.residual:
            v_res = self.res_conv(v)
            out = out + v_res

        # merge and combine heads
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)
        out = self.to_out(out)

        n = x.shape[1]
        out = out[:, -n:]

        return out

    def xforward(self, x, mask=None, lrp_params=None, verbose=False):
        """
        forward method for the explanation stage. it uses gamma_layers (go to utils_lrp) for linear layers.
        Args:
        :param x:
        :param mask:
        :param lrp_params: [Dictionary or None (default)] dic containing the necessary LRP parameters. None is
            equivalent to {'gamma': 0, 'eps': 1e-5, 'no_bias': False}. no_bias is True if the bias is discarded in
            LRP rules.
        :param verbose: (boolean)
        :return:
        """
        lrp_params = set_lrp_params(lrp_params)

        out, v = self.self_attention(
            x, mask=mask, xai_mode=True, lrp_params=lrp_params, verbose=verbose
        )
        # add depth-wise conv residual of values
        if self.residual:
            res_conv_ = modified_linear_layer(
                self.res_conv, lrp_params["gamma"], lrp_params["no_bias"]
            )
            v_res = res_conv_(v)
            out = out + v_res

        # merge and combine heads
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)
        to_out_ = modified_linear_layer(
            self.to_out[0], lrp_params["gamma"], lrp_params["no_bias"]
        )
        out = to_out_(out)

        n = x.shape[1]
        out = out[:, -n:]

        return out
