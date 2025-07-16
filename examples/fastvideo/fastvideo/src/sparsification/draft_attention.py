from math import sqrt
from typing import Literal, Union

from fastvideo.src.sparsification.block_sparse_attention import BlockSparseAttention

import mindspore.mint.nn.functional as F
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn, ops

__all__ = ["DraftAttention"]


class DraftAttention(nn.Cell):
    def __init__(
        self,
        head_dim: int,
        block_size: int,
        pool_h: int = 8,
        pool_w: int = 16,
        latent_h: int = 48,
        latent_w: int = 80,
        visual_len: int = 126_720,
        text_len: int = 0,  # we assume the text is at the end of the sequence, which is the case in the hunyuan model
        sparsity_ratio: float = 0.9,
        attn_type: Literal["flash", "block"] = "flash",
    ):
        super().__init__()

        self._block_size = block_size
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.visual_len = visual_len
        self.text_len = text_len
        self.sparsity_ratio = sparsity_ratio

        self.reorg_idx, self.restore_idx = self.generate_reorg_restore_indices(
            pool_h=pool_h, pool_w=pool_w, latent_h=latent_h, latent_w=latent_w, visual_len=visual_len, text_len=text_len
        )

        self._bmm = ops.BatchMatMul(transpose_b=True)

        self._use_block_attn = True if attn_type == "block" else False
        self.block_sparse_attn = BlockSparseAttention(head_dim, block_size)
        self.flash_attn = ops.flash_attention_score

    @staticmethod
    def generate_reorg_indices(
        total_len: int = 126_720, part_size: int = 640, block_size: int = 80, sub_block_size: int = 16
    ) -> list[int]:
        """
        Build a full-length reorder index list of length `total_len` that:
        1. Splits [0..total_len) into parts of size `part_size`.
        2. Within each part, splits into blocks of `block_size`.
        3. Within each block, splits into sub-blocks of `sub_block_size`.
        4. Reorders so that sub-blocks across all blocks are contiguous.

        Returns:
        reorg_idx: List[int] of length total_len, where
                    new_tensor = old_tensor[:, reorg_idx, ...]
        """
        assert total_len % part_size == 0, "total_len must be multiple of part_size"
        assert part_size % block_size == 0, "part_size must be multiple of block_size"
        assert block_size % sub_block_size == 0, "block_size must be multiple of sub_block_size"

        num_parts = total_len // part_size  # 126_720 // 640 = 198
        blocks_per_part = part_size // block_size  # 640 // 80 = 8
        subs_per_block = block_size // sub_block_size  # 80 // 16 = 5

        # build the pattern for one part
        part_pattern = []
        for c in range(subs_per_block):
            for b in range(blocks_per_part):
                start = b * block_size + c * sub_block_size
                part_pattern.extend(range(start, start + sub_block_size))
        assert len(part_pattern) == part_size

        # tile across all parts
        reorg_idx = []
        for p in range(num_parts):
            base = p * part_size
            reorg_idx.extend(base + i for i in part_pattern)

        return reorg_idx

    def generate_reorg_restore_indices(
        self,
        pool_h: int = 8,
        pool_w: int = 16,
        latent_h: int = 48,
        latent_w: int = 80,
        visual_len: int = 126_720,
        text_len: int = 0,
        # if there is text at the behind, the text_len should be added to the reorg and restore indices without changing the order
    ) -> tuple[list[int], list[int]]:
        """
        Build the reorg and restore indices in one go.
        """
        part_size = latent_w * pool_h
        block_size = latent_w
        sub_block_size = pool_w

        assert latent_h % pool_h == 0, "latent_h must be multiple of pool_h"
        assert visual_len % part_size == 0, "total_len must be multiple of part_size"
        assert block_size % sub_block_size == 0, "block_size must be multiple of sub_block_size"

        reorg_idx = self.generate_reorg_indices(visual_len, part_size, block_size, sub_block_size)

        # invert the mapping
        restore_idx = [0] * visual_len
        for new_pos, orig_pos in enumerate(reorg_idx):
            restore_idx[orig_pos] = new_pos

        # add the text_len to the reorg and restore indices
        if text_len > 0:
            reorg_idx += [i for i in range(visual_len, text_len + visual_len)]
            restore_idx += [i for i in range(visual_len, text_len + visual_len)]

        return reorg_idx, restore_idx

    def sample_qk_attention_2d(
        self,
        q: Tensor,
        k: Tensor,
        frame_h: int,
        frame_w: int,
        pool_h: int,
        pool_w: int,
    ) -> Tensor:
        """
        q, k: [L, H, D] where the first num_frames*frame_h*frame_w tokens are video,
                L_vid = num_frames * frame_h * frame_w.
        frame_h, frame_w: spatial dims of each frame.
        pool_h, pool_w: 2D pooling kernel (and stride) for sampling each frame.

        Returns:
        attn_map: [H, S, S] where
            S = num_frames * ceil(frame_h/pool_h) * ceil(frame_w/pool_w).
        """
        assert (
            len(q.shape) == 3
        ), "q must be of shape [L, H, D], similar for k, which is similar as flash attention input."
        L, H, D = q.shape
        frame_tokens = frame_h * frame_w
        assert L % frame_tokens == 0, "L must be multiple of frame_h*frame_w"
        num_frames = L // frame_tokens

        # 1) Slice out the video part and reshape to frames:
        #    [L, H, D] → [num_frames, frame_h, frame_w, H, D]
        q_vid = q.view(num_frames, frame_h, frame_w, H, D)
        k_vid = k.view(num_frames, frame_h, frame_w, H, D)

        # 2) Permute & merge (num_frames, H*D) into channel dim:
        #    → [num_frames, H*D, frame_h, frame_w]
        q_vid = q_vid.permute(0, 3, 4, 1, 2).reshape(num_frames, H * D, frame_h, frame_w)
        k_vid = k_vid.permute(0, 3, 4, 1, 2).reshape(num_frames, H * D, frame_h, frame_w)

        # 3) 2D max‐pool each frame (ceil_mode ensures we cover the edges):
        #    → [num_frames, H*D, S_h, S_w]
        q_pooled = F.avg_pool2d(q_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)
        k_pooled = F.avg_pool2d(k_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)

        S_h, S_w = q_pooled.shape[-2:]
        S = num_frames * S_h * S_w

        # 4) Un‐merge channel back to [S, H, D]:
        #    → [num_frames, H, D, S_h, S_w] → [S, H, D]
        sampled_q = q_pooled.reshape(num_frames, H, D, S_h, S_w).permute(0, 3, 4, 1, 2).reshape(S, H, D)
        sampled_k = k_pooled.reshape(num_frames, H, D, S_h, S_w).permute(0, 3, 4, 1, 2).reshape(S, H, D)

        # 5) Compute per‐head scaled dot‐prod attention:
        #    [S, H, D] → [H, S, D]
        q_heads = sampled_q.permute(1, 0, 2)
        k_heads = sampled_k.permute(1, 0, 2)

        # → [H, S, S]
        scores = self._bmm(q_heads, k_heads) / sqrt(D)
        attn_map = mint.softmax(scores, dim=-1)

        return attn_map

    def attention_percentile_mask_headwise(self, attn_map: Tensor, r: float) -> Tensor:
        """
        Build a mask per head so that each head keeps its top-r fraction of entries as True.

        Args:
        attn_map: Tensor of shape [H, S, S], attention scores (e.g. after softmax).
        r: float in (0,1), fraction of entries *per head* to keep True.

        Returns:
        mask: BoolTensor of shape [H, S, S], where for each head h,
                mask[h].float().mean() ≈ r.
        """
        H, S, _ = attn_map.shape
        mask = mint.zeros_like(attn_map, dtype=mstype.bool_)

        # process each head independently
        for h in range(H):
            head_scores = attn_map[h]  # [S, S]
            flat = head_scores.flatten()  # [S*S]
            n = flat.numel()
            k = int((1.0 - r) * n)  # number of smallest to exclude

            if k == 0:
                mask[h] = True
                continue
            if k >= n:
                # nothing to keep
                continue

            # threshold = max of the k smallest scores
            threshold = mint.topk(flat, k, largest=False)[0].max()

            # build head mask
            mask[h] = head_scores >= threshold

        return mask

    def construct(
        self,
        q,
        k,
        v,
        batch_size=1,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        force_dense_attn: bool = False,
        return_blockmask: bool = False,
        **kwargs,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        # ======== block_sparse_attention setup ========
        q_len, num_heads, head_dim = q.shape
        # Xuan: in 'head_mask_type':
        #   Tensor([x] * 24, dtype=torch.int32) (PS: in hunyuan, there are 24 heads)
        # mask_type = 0 denotes the dense attention
        # mask_type = -1 denotes the streaming attention
        # mask_type = 1 denotes the blocksparse attention (we use this here)
        # ======== block_sparse_attention setup ========

        base_blockmask = None
        if not force_dense_attn:
            attn = self.sample_qk_attention_2d(
                q[: self.visual_len],
                k[: self.visual_len],
                frame_h=self.latent_h,
                frame_w=self.latent_w,
                pool_h=self.pool_h,
                pool_w=self.pool_w,
            )
            q_block_num = (q_len + self._block_size - 1) // self._block_size
            k_block_num = (k.shape[0] + self._block_size - 1) // self._block_size
            base_blockmask_visual = self.attention_percentile_mask_headwise(attn, 1 - self.sparsity_ratio)
            base_blockmask = mint.full((num_heads, q_block_num, k_block_num), fill_value=True, dtype=mstype.bool_)
            base_blockmask[
                :, : base_blockmask_visual.shape[1], : base_blockmask_visual.shape[2]
            ] = base_blockmask_visual

        # re-organize the q and k for visual alignment  # todo
        # q = q[self.reorg_idx, :, :]
        # k = k[self.reorg_idx, :, :]
        # v = v[self.reorg_idx, :, :]

        if self._use_block_attn and not force_dense_attn:
            H, D = q.shape[-2:]
            q = q.reshape(batch_size, -1, H, D).swapaxes(1, 2).reshape(batch_size * H, -1, D)
            k = k.reshape(batch_size, -1, H, D).swapaxes(1, 2).reshape(batch_size * H, -1, D)
            v = v.reshape(batch_size, -1, H, D).swapaxes(1, 2).reshape(batch_size * H, -1, D)

            x = self.block_sparse_attn(q, k, v, base_blockmask)

            x = x.reshape(batch_size, H, -1, D).swapaxes(1, 2).reshape(-1, H, D)
        else:
            H, D = q.shape[-2:]

            q, k, v = q.reshape(batch_size, -1, H, D), k.reshape(batch_size, -1, H, D), v.reshape(batch_size, -1, H, D)
            if base_blockmask is not None:
                base_blockmask = mint.repeat_interleave(
                    mint.repeat_interleave(~base_blockmask, self._block_size, dim=-1), self._block_size, dim=-2
                )[None]

            x = self.flash_attn(
                q,
                k,
                v,
                head_num=H,
                attn_mask=base_blockmask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scalar_value=1 / sqrt(D),
                input_layout="BSND",
                sparse_mode=1,
            )

            x = x.reshape(-1, H, D)

        # re-organize the x to the original order
        # x = x[self.restore_idx, :, :]  # todo

        # x with shape [(bxs), a, d]
        x = x.view(batch_size, q_len, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]

        if return_blockmask:
            return x, base_blockmask
        return x
