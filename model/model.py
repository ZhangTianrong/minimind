import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# === Normalization ===
# Normalization, in the context of deep learning, refers to centering (to get zero mean) and/or scaling (to get unit variance) of features to prevent gradients from
# vanishing or exploding during training, so that models can converge faster and more reliably. (We will see later that choosing a non-saturating activation
# function is also part of the solution to the vanishing gradient problem. See also [this blog](https://blog.csdn.net/qq_44091004/article/details/129440577).)
# The most common normalization methods include [Batch Normalization](https://arxiv.org/abs/1502.03167), [Layer Normalization](https://arxiv.org/abs/1607.06450v1),
# [Instance Normalization](https://arxiv.org/abs/1607.08022), and [Group Normalization](https://arxiv.org/abs/1803.08494), which differenciate in how they interpret
# feature and over which coordinates of their data the normalization is performed, as illustrated in the following figure from Group Normalization:
# ![BN vs LN vs IN vs GN](https://ar5iv.labs.arxiv.org/html/1803.08494/assets/x2.png)
# There have been a lot of tutorials and discussions on these normalization methods, but most of them take the image perspective using inputs of the shape 
# `(batch_size, height_times_width, channels)` as an example. While it is tempting to analogize `height_times_width` to `seq_len` and `channels` to `hidden_dim`,
# it can actually lead to confusion and misunderstandings if we just rename the coordinates, as we will see when we come to, for example, LN.
# Consider a typical textual input tensor $x$ with the shape of `(batch_size, seq_len, hidden_dim)` (i.e. the coordinates stand for `sample_idx` (in the batch),
# `position_idx` (of a token in the sequence) and `feature_idx` respectively):
# + Batch Normalization treats the input as batched features and computes statistics across the batching dimension. If the `[seq_len, hidden_dim]` slices of $x$ are
#   treated as the features. Then for all `postion_idx, feature_idx` pairs, the means and variances are found over the `[batch_size]` slices along the only
#   coordinate left, namely `sample_idx`. However, as we will see in LN, `feature_idx` has its intrinsic connections with the operations of tranformer models to be 
#   considered the natural representative of what features are. So typically, `[sample\_idx, position\_idx]` are collectively regarded as the batch dimension, and
#   the reduction happens to both of them, i.e.
#   $$\operatorname{BN}(x) = \frac{x - \operatorname{mean}(x,\operatorname{reduce\_dim}=[{\tt sample\_idx}, {\tt position\_idx}])}{\sqrt{\operatorname{var}(x,\operatorname{reduce\_dim}=[{\tt sample\_idx}, {\tt position\_idx}])+\epsilon}}\odot\gamma+\beta$$
#   The batch-dependant statistics are computed with each mini-batch during training, and the running statistics are stored for use during inferencing (with variance
#   estimated using the biased estimator). As a result, BN is known to perform poorly at small batch sizes and can suffer from distribution shift at test time. The
#   normalization across tokens within a sequance also makes it less suitable in text domain where the token ordering is important.
# + Layer Normalization interpret features as whatever the preceeding layer of the model processes (thus the name) and carries out normalization over the feature
#   dimensions. In the case of Transformer models, both attention mechanisms and linear layers operates on the last coordinate. Hence it treats every `[hidden_dim]`
#   slice of $x$ as a feature vector and computes means and variances for each slice, i.e.
#   $$\operatorname{LN}(x) = \frac{x - \operatorname{mean}(x,\operatorname{reduce\_dim}=[{\tt feature\_idx}])}{\sqrt{\operatorname{var}(x,\operatorname{reduce\_dim}=[{\tt feature\_idx}])+\epsilon}}\odot\gamma+\beta$$
#   with the coordinates reduced over the exact complement of those of BN. In the case of CNNs of the image domain, however, since convolutional layers operate on
#   `[height_times_width, channels]` slices, each of such slices is regarded as a feature vector in LN; so the reduction is performed over both `pixel_idx` and
#   `channel_idx` coordinates, overlapping with the typical image BNs at `pixel_idx` due to a different choice of feature dimensions (i.e. `channel_idx` alone). If
#   we simply rename the to adapt the LN from image to text domain, we would be computing means and variances for `[seq_len, hidden_dim]` slices of $x$ instead,
#   which diverges from the actual practices.
# + Instance Normalization regards the input as batched instances such that each instance corresponds to features that minimally represent a sample. In the case of
#   Transformer models, the `hidden_dim` `[seq_len]` slices of a sample in $x$ can be treated as `hidden_dim` instances of it since each of those slices potentially
#   rerpesents a different interpretation of the entire text sequence. Therefore, the reduction is performed over the `position_idx` coordinate, i.e.
#   $$\operatorname{IN}(x) = \frac{x - \operatorname{mean}(x,\operatorname{reduce\_dim}=[{\tt position\_idx}])}{\sqrt{\operatorname{var}(x,\operatorname{reduce\_dim}=[{\tt position\_idx}])+\epsilon}}\odot\gamma+\beta$$
#   IN is known to be used in style transfer in the image domain, where the style is considered to be dominated by only a few instances of the image, but as it
#   normalizes over the `position_idx` coordinate, it is not ideal for texts for the same reason as BN.\
# + Group Normalization is a generalization of LN or IN that allows the user to group features or instances into groups which share the same statistics. For example,
#   instead of computing means and variances for each `[hidden_dim]` slice of $x$ in LN, we can group them into `num_groups` groups and compute the statistics for
#   each `[hidden_dim/num_groups]` sub-slice of $x$ in GN. 
#   TODO: See why GN is not favored in Transformer models. Since multi-head attention in Transformer models is already a form of feature grouping, GN seems to be
#   a natural choice.
# == RMSNorm ==
# [RMSNorm](https://arxiv.org/abs/1910.07467) is a variant of LN that omits the step of centering (and drops $\beta$), i.e.
# $$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\operatorname{var}(x,\operatorname{reduce\_dim}=[{\tt position\_idx}, {\tt feature\_idx}])+\epsilon}}\odot\gamma$$
# TODO: not finished yet, why rmsnorm over layernorm? 1) no centering, 2) no bias
# The invariance properties of RMSNorm vs the other normalization methods are summarized in [this table](https://ar5iv.labs.arxiv.org/html/1910.07467#S4.T1).
# PyTorch implemented RMSNorm as used in `torch.nn.RMSNorm` and `torch.nn.functional.rms_norm` with `rms_norm_symint` in C++ [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp)
# using the following interface more or less the same as LN:
# + `input`: The tensor $x$ to be normalized
# + `normalized_shape`: The shape of the dimensions to reduce over, i.e. the sizes of `reduce_dim` in the above formulas or `dim` in PyTorch functions like
#   `torch.mean`. For `input` with shape $(d_1,\cdots,d_n)$, the acceptable dimensions need to be continuous and start from the last dimension, e.g. `[n-1, n]`
#   with shape $[d_{n-1},d_n]$ or `n` with shape $d_n$.
# + `weight`: The learnable parameter $\gamma$ of `normalized_shape`. In `torch.nn.RMSNorm`, It can be disabled by setting `learnable_affine` to `False`.
# + `eps`: The $\epsilon$ in the above formulas to prevent division by zero. When it is `None`, it defaults to the machine epsilon [here](https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon).

# Option to use the PyTorch implementation of RMSNorm (not available with the default PyTorch version used by MiniMind)
try:
    # TODO: The interface of RMSNorm implemented in MiniMind (identical to [Meta's implementation](https://github.com/meta-llama/llama/blob/main/llama/model.py)) is 
    # not compatible with the PyTorch implementation. We need to edit the code to ensure that the implementations can be used interchangeably. For now, we are 
    # adjusting the PyTorch implementation to match the MiniMind implementation.
    from torch.nn import RMSNorm as RMSNormPT
    class RMSNorm(RMSNormPT):
        def __init__(self, dim: int, eps: float):
            super().__init__(normalized_shape=[dim], eps=eps)
except ImportError:
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim)) # `gamma` of the same shape as features

        def _norm(self, x):
            # Replicating the C++ implementation below:
            # ```cpp
            #   Tensor upcasted_input = input.to(opmath_t);
            #   Tensor rqrst_input = rsqrt(at::pow(upcasted_input, 2).mean(dims_to_reduce_ref, /*keep_dim=*/true).add_(eps_val));
            #   Tensor result = upcasted_input.mul(rqrst_input).type_as(input);
            # ```
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # The function `rsqrt` computes reciprocal square root with e.g. CUDA's `rsqrtf`
                                                                                # function when available.

        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight

# === Positional Encoding ===
def precompute_pos_cis(
    dim: int,               # Hidden dimension
    end: int,               # Maximum sequence length
    theta: float = 10000.0  # Choosing base frequency of 10000^{-2i/d} by default
):                                                                                      
    freqs = (1.0 / # (dim//2,)                                                          # Rotray Positional Embedding
        (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)))               # It is suggested to refer to the [blogs](https://kexue.fm/tag/rope) by the
    t = torch.arange(end, device=freqs.device) # type: ignore # (end,)                  # author of [RoFormer](https://arxiv.org/abs/2104.09864) for a more detailed
    freqs = torch.outer(t, freqs).float() # type: ignore # (end, dim//2)                # explanation of the positional encoding.
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 # (end, dim//2)   # But in short, RoPE is an absolute positional encoding that captures the
    return pos_cis                                                                      # relative positions through attention, i.e.
                                                                                        # $$ (\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k} $$
                                                                                        # where $m,n$ represent the position of token in the query and the key
                                                                                        # respectively and $\boldsymbol{\mathcal{R}}_m \boldsymbol{q}, \boldsymbol{\mathcal{R}}_n \boldsymbol{k}$
                                                                                        # refer to the operation of applying RoPE to these tokens' query/key 
                                                                                        # vectors, e.g.
                                                                                        # $$\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} = \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}$$

def apply_rotary_emb(
    xq,     # (bs, seq_len, hidden_dim)
    xk,     # (bs, seq_len, hidden_dim)
    pos_cis # = pos_cis[current_idx:current_idx + seq_len] # (seq_len, hidden_dim//2)
):
    def unite_shape(pos_cis, x): # x is viewd as complex
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # bs, seq_len, *, hidden_dim//2 
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (bs, seq_len, hidden_dim//2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # (bs, seq_len, hidden_dim//2)
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # Multi-Head Attention vs. Grouped-Query Attention vs. Multi-Query Attention
        assert args.n_heads % self.n_kv_heads == 0                                      # The following figure from [GQA](https://arxiv.org/abs/2305.13245) 
                                                                                        # intuitively demonstrates the difference between the three types of attention:
                                                                                        # ![overview of mh, gqa, and mqa](https://arxiv.org/html/2305.13245v3/extracted/5314337/images/gmq_architecture.png)
                                                                                        # The use of heads in Transformer models was proposed to mitigate the lack
                                                                                        # of discriptive power of single-head attention in [Attention is All You Need](https://arxiv.org/abs/1706.03762),
                                                                                        # because in MHA, each head can use its own Softmax to allow for different
                                                                                        # ways tokens attend to each other, which is frequently analogized to
                                                                                        # different kernels in a CNN in that they both operate on the same feature
                                                                                        # space using the same architecture but with different parameters adopted.
                                                                                        # In practice, changes to the number of heads are achived by changing how
                                                                                        # many partitions wq, wk, and wv are split into, while preserving the total
                                                                                        # number of parameters. Empically speaking, the use of MHA achieves better
                                                                                        # performance with the same number of parameters.
                                                                                        # TODO: How to choose the number of heads in MHA?
                                                                                        # While MHA ensures that the query of each head has its own key and value,
                                                                                        # GQA and MQA allow for queries of different heads to share keys and values,
                                                                                        # with MQA being a special case of GQA where all query heads share the same
                                                                                        # keys and values. Such a design trades descriptiveness for memory use, and
                                                                                        # the choice of n_rep = n_heads/n_kv_heads balances the two. In practice, 
                                                                                        # GQA and MQA keeps the size of wq but reduces the size of wk and wv. GQA
                                                                                        # and MQA. Hence, here we repeate wk and wv n_rep times before feeding them
                                                                                        # to the attention implementation.
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.k_cache, self.v_cache = None, None
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)    # Buffer are typically used for module states that are not trainable,
                                                                # e.g. running statistics of a batch normalization layer.

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 更高效的kv_cache实现
        if kv_cache and self.eval():
            if seqlen == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)):
                xk = torch.cat((self.k_cache, xk), dim=1)
                xv = torch.cat((self.v_cache, xv), dim=1)
            self.k_cache, self.v_cache = xk, xv

        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash and seqlen != 1:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)
        ])

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x, pos_cis, kv_cache=False):
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)   # Pre-Norm vs. Post-Norm
        out = h + self.feed_forward(self.ffn_norm(h))                       # Pre-Norm: $\boldsymbol{x}_{t+1}=\boldsymbol{x}_t+F_t(\mathrm{Norm}(\boldsymbol{x}_t))$
        return out                                                          # Post-Norm: $boldsymbol{x}_{t+1}=\mathrm{Norm}(\boldsymbol{x}_t+F_t(\boldsymbol{x}_t))$
                                                                            # Empirically Pre-Norm allows for faster convergence but Post-Norm yields better
                                                                            # results.
                                                                            # [The blog here](https://kexue.fm/archives/9009) discusses the reasons behind this 
                                                                            # phenomenon. Briefly speaking, when the Transformer model deepens, Pre-Norm leads to
                                                                            # significantly smaller differences in activations between layers, which in a sense,
                                                                            # resembles widening the network more than deepening it, hurting the expressiveness of 
                                                                            # the model (while improving memorization).


class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LMConfig = None):
        super().__init__(params)
        if not params:
            params = LMConfig()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                kv_cache=False, **keyargs):
        current_idx = 0
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        pos_cis = self.pos_cis[current_idx:current_idx + seqlen]
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                             ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True):
        # rp: repetition_penalty
        index = idx.shape[1]
        init_inference = True
        while idx.shape[1] < max_new_tokens - 1:
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)

            logits = inference_res.logits
            logits = logits[:, -1, :]

            for token in set(idx.tolist()[0]):
                logits[:, token] /= rp

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            if idx_next == eos:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            if stream:
                yield idx[:, index:]

        if not stream:
            yield idx[:, index:]

    @torch.inference_mode()
    def eval_answer(self, idx):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        inference_res = self(idx_cond)
        logits = inference_res.logits
        logits = logits[:, -1, :]
        return logits
