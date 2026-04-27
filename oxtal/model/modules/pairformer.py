# pylint: disable=C0114
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from openfold.model.dropout import DropoutRowwise
from openfold.model.outer_product_mean import OuterProductMean  # Alg 9 in AF3
from openfold.model.primitives import LayerNorm
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,  # Alg 13 in AF3
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,  # Alg 12 in AF3
)
from openfold.utils.checkpointing import checkpoint_blocks
from oxtal.model.modules.primitives import LinearNoBias, Transition
from oxtal.model.modules.transformer import AttentionPairBias
from oxtal.model.utils import sample_msa_feature_dict_random_without_replacement


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        dropout: float = 0.25,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
                Defaults to 128.
            c_hidden_pair_att (int, optional): hidden dim [for TriangleAttention]. Defaults to 32.
            no_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
            dropout (float, optional): dropout ratio [for TriangleUpdate]. Defaults to 0.25.
        """
        super().__init__()
        self.n_heads = n_heads
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.pair_transition = Transition(c_in=c_z, n=4)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        if inplace_safe:
            z = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.tranpose(-1, -2) if pair_mask is not None else None,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z)
            if self.c_s > 0:
                s += self.attention_pair_bias(
                    a=s,
                    s=None,
                    z=z,
                )
                s += self.single_transition(s)
            return s, z
        else:
            tmu_update = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            tmu_update = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            z = z + self.dropout_row(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)
            z = z + self.dropout_row(
                self.tri_att_end(
                    z,
                    mask=pair_mask.tranpose(-1, -2) if pair_mask is not None else None,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)

            z = z + self.pair_transition(z)
            if self.c_s > 0:
                s = s + self.attention_pair_bias(
                    a=s,
                    s=None,
                    z=z,
                )
                s = s + self.single_transition(s)
            return s, z


class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks [for PairformerStack]. Defaults to 48.
            n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            dropout (float, optional): dropout ratio. Defaults to 0.25.
            blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        if self.n_blocks < 0:
            print("Disabling pairformer")
            return
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = PairformerBlock(n_heads=n_heads, c_z=c_z, c_s=c_s, dropout=dropout)
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the update of s and z
                [..., N_token, c_s]
                [..., N_token, N_token, c_z]
        """
        if self.n_blocks < 0:
            return torch.zeros_like(s), torch.zeros_like(z)
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z


class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for TemplateEmbedder. Defaults to 2.
            c (int, optional): hidden dim of TemplateEmbedder. Defaults to 64.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            dropout (float, optional): dropout ratio for PairformerStack. Defaults to 0.25.
                Note this value is missed in Algorithm 16, so we use default ratio for Pairformer
            blocks_per_ckpt: number of TemplateEmbedder/Pairformer blocks in each activation
                checkpoint Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "b_template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "b_template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0
        # Disable all initialization
        if self.n_blocks is None:
            return

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values()) + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,  # pylint: disable=W0613
        pair_mask: torch.Tensor = None,  # pylint: disable=W0613
        use_memory_efficient_kernel: bool = False,  # pylint: disable=W0613
        use_deepspeed_evo_attention: bool = False,  # pylint: disable=W0613
        use_lma: bool = False,  # pylint: disable=W0613
        inplace_safe: bool = False,  # pylint: disable=W0613
        chunk_size: Optional[int] = None,  # pylint: disable=W0613
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]): input feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor, optional): pair masking. Default to None.
                [..., N_token, N_token]

        Returns:
            torch.Tensor: the template feature
                [..., N_token, N_token, c_z]
        """
        # In this version, we do not use TemplateEmbedder by setting n_blocks=0
        if "template_restype" not in input_feature_dict or self.n_blocks < 1:
            return 0
        return 0
