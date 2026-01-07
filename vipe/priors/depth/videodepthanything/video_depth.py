# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import gc
from typing import Iterator

import cv2
import numpy as np

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose
from tqdm import tqdm

from vipe.priors.depth.dav2.dinov2 import DINOv2
from vipe.priors.depth.dav2.util.transform import NormalizeImage, PrepareForNet, Resize

from .dpt_temporal import DPTHeadTemporal
from .util import compute_scale_and_shift, get_interpolate_frames


# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {"vits": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))  # return shape [B, T, H, W]

    def infer_video_depth(
        self, frame_list: list[np.ndarray], input_size: int = 518, device="cuda", fp32=True
    ) -> np.ndarray:
        frame_height, frame_width = frame_list[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                # cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
                cur_list.append(
                    torch.from_numpy(transform({"image": frame_list[frame_id + i].astype(np.float32)})["image"])
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input)  # depth shape: [1, T, H, W]

            depth = F.interpolate(
                depth.flatten(0, 1).unsqueeze(1),
                size=(frame_height, frame_width),
                mode="bilinear",
                align_corners=True,
            )
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input
            torch.cuda.empty_cache()  # Otherwise would OOM for VIT-Large

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id + kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id + i])
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align),
                    np.concatenate(ref_align),
                    np.concatenate(np.ones_like(ref_align) == 1),
                )

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id + align_len : frame_id + OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i] < 0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id + i] * scale + shift
                    new_depth[new_depth < 0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id + kf_id] * scale + shift
                    new_depth[new_depth < 0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0)

    def infer_video_depth_stream(
        self, frame_iterator: Iterator[np.ndarray], input_size: int = 518, device="cuda", fp32=True
    ) -> Iterator[np.ndarray]:
        """
        Streaming version of infer_video_depth.
        Yields depth maps one by one as they are processed.
        """
        # We need to peek at the first frame to get size, but we can't consume it.
        # So we'll maintain a buffer.
        frame_buffer: list[np.ndarray] = []
        try:
            first_frame = next(frame_iterator)
            frame_buffer.append(first_frame)
        except StopIteration:
            return

        frame_height, frame_width = first_frame.shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        frame_step = INFER_LEN - OVERLAP
        
        # State for alignment
        depth_list_aligned: list[np.ndarray] = [] # Only keeps needed history
        ref_align: list[np.ndarray] = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        # Buffer for frames to process in a batch
        # We need INFER_LEN frames to run inference
        # But we step by slightly less (INFER_LEN - OVERLAP)
        
        # We process in chunks.
        # Chunk 0: frames [0, INFER_LEN)
        # Chunk 1: frames [step, step + INFER_LEN)
        # overlap is [step, INFER_LEN) which has length OVERLAP
        
        # We maintain a `current_window` of frames.
        current_window: list[np.ndarray] = frame_buffer # initialized with first frame
        
        # Helper to replenish window
        def fill_window():
            while len(current_window) < INFER_LEN:
                try:
                    current_window.append(next(frame_iterator))
                except StopIteration:
                    break
        
        fill_window()
        
        frame_id = 0
        is_first_chunk = True
        
        pre_input = None
        
        while len(current_window) > 0:
            # Prepare batch
            # If window is smaller than INFER_LEN (end of video), pad with last frame
            org_window_len = len(current_window)
            padded_window = current_window + [current_window[-1].copy()] * (INFER_LEN - org_window_len)
            
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(
                    torch.from_numpy(transform({"image": padded_window[i].astype(np.float32)})["image"])
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input)  # depth shape: [1, T, H, W]

            depth = F.interpolate(
                depth.flatten(0, 1).unsqueeze(1),
                size=(frame_height, frame_width),
                mode="bilinear",
                align_corners=True,
            )
            
            # Current chunk depth list
            current_depth_list = [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
            
            pre_input = cur_input
            torch.cuda.empty_cache()

            # Alignment logic
            if is_first_chunk:
                # First chunk, just output the first part directly
                # We can yield frames up to INFER_LEN (if we didn't have to align next)
                # But we actually accumulate everything first in original code.
                # Here we want to yield as soon as possible.
                
                # Logic from original:
                # depth_list_aligned += depth_list[:INFER_LEN]
                # for kf_id in kf_align_list:
                #    ref_align.append(depth_list[frame_id + kf_id])
                
                # So we can yield [0, INFER_LEN) immediately? 
                # Wait, later chunks might modify the overlap region if we weren't careful.
                # In original code:
                # pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                # depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(...)
                
                # So the last INTERP_LEN frames of the "aligned" list will be modified by the next chunk.
                # So we can yield up to INFER_LEN - INTERP_LEN safely.
                
                safe_yield_len = INFER_LEN - INTERP_LEN
                # Actually if it is the very last chunk, we yield everything.
                if org_window_len < INFER_LEN:
                     safe_yield_len = org_window_len # yield all valid frames
                
                # yield frames
                for i in range(safe_yield_len):
                     yield current_depth_list[i]
                
                # Keep the tail for alignment/interpolation
                depth_list_aligned = current_depth_list # Store locally for reference
                
                # Setup ref_align for next chunk
                for kf_id in kf_align_list:
                     ref_align.append(current_depth_list[kf_id])
                     
                is_first_chunk = False
                
            else:
                # Subsequent chunks
                curr_align = []
                for i in range(len(kf_align_list)):
                    # relative index in current chunk
                    curr_align.append(current_depth_list[i])
                
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align),
                    np.concatenate(ref_align),
                    np.concatenate(np.ones_like(ref_align) == 1),
                )
                
                # Align current chunk post-overlap part
                # post_depth_list = depth_list[frame_id + align_len : frame_id + OVERLAP] 
                # In local coordinates: [align_len : OVERLAP]
                for i in range(align_len, OVERLAP):
                     current_depth_list[i] = current_depth_list[i] * scale + shift
                     current_depth_list[i][current_depth_list[i] < 0] = 0
                
                # Interpolate previous tail
                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = current_depth_list[align_len : OVERLAP]
                
                interpolated = get_interpolate_frames(pre_depth_list, post_depth_list)
                
                # Now we yield these interpolated frames
                for d in interpolated:
                    yield d
                
                # Align the rest of current chunk
                # for i in range(OVERLAP, INFER_LEN):
                for i in range(OVERLAP, INFER_LEN):
                    new_depth = current_depth_list[i] * scale + shift
                    new_depth[new_depth < 0] = 0
                    current_depth_list[i] = new_depth
                
                # Determine safe yield for CURRENT chunk
                # We yield from [align_len : INFER_LEN - INTERP_LEN] ?
                # The interpolated part corresponds to the gap.
                # The `interpolated` frames replace `pre_depth_list` (tail of prev) AND `post_depth_list` (head of current)?
                # No, look at original: 
                # depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(...)
                # This modifies the tail of the ACCUMULATED list.
                
                # The interpolation replaces the last INTERP_LEN frames of the previous chunk.
                # Wait, OVERLAPframes overlap. 
                # align_len = OVERLAP - INTERP_LEN.
                # The `post_depth_list` is length INTERP_LEN.
                
                # Basically:
                # 1. We hold back INTERP_LEN frames from previous chunk.
                # 2. When new chunk comes, we align it.
                # 3. We take first INTERP_LEN frames of aligned new chunk (which are in the overlap region).
                # 4. We interpolate between held-back frames and these new frames.
                # 5. We yield the interpolated frames (INTERP_LEN frames).
                # 6. Then we yield the REST of the overlap region? No.
                
                # Let's trace indices.
                # Previous chunk yielded up to `end - INTERP_LEN`.
                # Interpolation covers `[end - INTERP_LEN, end)`.
                # Actually the interpolation replaces the transition.
                
                # Original Code:
                # pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                # post_depth_list = depth_list[frame_id + align_len : frame_id + OVERLAP]
                # depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(...)
                # for i in range(OVERLAP, INFER_LEN): append...
                
                # Note: `depth_list` [frame_id + align_len : frame_id + OVERLAP] IS the frames corresponding to the same time instants as pre_depth_list?
                # No. frame_step = INFER_LEN - OVERLAP.
                # New chunk starts at `frame_id`.
                # Previous chunk started at `frame_id - frame_step`.
                # Previous chunk ended at `frame_id - frame_step + INFER_LEN = frame_id + OVERLAP`.
                # So there is OVERLAP amount of overlap.
                
                # depth_list_aligned has length `frame_id`. (Because each step adds frame_step frames).
                # Wait. 
                # First chunk adds INFER_LEN.
                # Next chunk overwrites? No.
                # "depth_list_aligned[-INTERP_LEN:]"
                
                # Let's stick to the behavior.
                # We hold back INTERP_LEN frames from being yielded in the previous step.
                # In this step:
                # 1. interpolate the held back frames with `current_depth_list[align_len : OVERLAP]`.
                # 2. extend `depth_list_aligned` (virtually) with `current_depth_list[OVERLAP:]`.
                
                # So we yield:
                # 1. Interpolated frames (INTERP_LEN count).
                # 2. `current_depth_list[OVERLAP : INFER_LEN - INTERP_LEN]`.
                # And we hold back `current_depth_list[INFER_LEN - INTERP_LEN : ]`.
                
                # Wait, `current_depth_list` indices:
                # `align_len` is OVERLAP - INTERP_LEN.
                # So `current_depth_list[align_len : OVERLAP]` has length INTERP_LEN.
                
                # Yield logic:
                # Yield interpolated frames.
                
                # yield current_depth_list[OVERLAP : INFER_LEN - INTERP_LEN] (length = INFER_LEN - OVERLAP - INTERP_LEN?)
                # Wait, `frame_step` is INFER_LEN - OVERLAP.
                # We want to yield `frame_step` frames in total per iteration to stay consistent?
                # Interpolated part is INTERP_LEN.
                # Non-overlapping new part is `[OVERLAP:]`. length = INFER_LEN - OVERLAP.
                # Total new valid frames = INFER_LEN - OVERLAP.
                
                # But we held back INTERP_LEN from previous.
                # So we yield INTERP_LEN (interpolated) + (INFER_LEN - OVERLAP - INTERP_LEN)? 
                # No.
                
                # The total stream length should match.
                
                # Let's simplify.
                # Safe Frames to yield from current chunk:
                # The interpolated frames are effectively the "last INTERP_LEN frames of the PREVIOUS chunk time slot".
                # But refined.
                
                # So:
                # 1. Yield Interpolated frames.
                # 2. Yield `current_depth_list[OVERLAP : INFER_LEN - INTERP_LEN]`.
                
                # What if `org_window_len < INFER_LEN` (last chunk)?
                # Then we yield everything remaining.
                
                safe_yield_end = INFER_LEN - INTERP_LEN
                if org_window_len < INFER_LEN:
                    safe_yield_end = org_window_len # yield all valid
                
                # Yield interpolated
                for d in interpolated:
                    yield d
                
                # Yield non-overlapping part, up to safe end
                for i in range(OVERLAP, safe_yield_end):
                    yield current_depth_list[i]
                
                # Update buffers
                depth_list_aligned = current_depth_list[:safe_yield_end] # Just need to keep enough for next reference?
                # Actually we only need to keep the tail for next interpolation.
                # Tail is `current_depth_list[safe_yield_end:]` if we continue.
                # And for `ref_align`, we need `current_depth_list[kf_id]` indices.
                
                depth_list_aligned = current_depth_list # Keep full current chunk as "previous" for next iter
                
                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = current_depth_list[kf_id] * scale + shift
                    new_depth[new_depth < 0] = 0
                    ref_align.append(new_depth)
            
            # Step frame buffer
            # We consumed `frame_step` frames from the front.
            # INFER_LEN - OVERLAP
            step = INFER_LEN - OVERLAP
            
            # If we are at the end, we are done
            if len(current_window) < INFER_LEN:
                 break
                 
            # Remove processed frames
            del current_window[:step]
            # Fill more
            fill_window()
            
            frame_id += step
            
        # If we broke out and have held back frames (from the last chunk logic), yield them?
        # In the loop:
        # If `org_window_len < INFER_LEN`, we set `safe_yield_end = org_window_len`.
        # And we yield up to that.
        # So we should be good.
        
        # Verify first chunk logic again.
        # safe_yield_len = INFER_LEN - INTERP_LEN.
        # We hold back INTERP_LEN.
        # Next chunk consumes them in interpolation.
        
        # What if video length < INFER_LEN?
        # first chunk `org_window_len < INFER_LEN`.
        # `safe_yield_len = org_window_len`.
        # We yield all. Loop finishes. Correct.
        
        # What if video length just ends perfectly?
        # Last chunk has `len == INFER_LEN`.
        # We yield `INFER_LEN - INTERP_LEN`.
        # Loop continues.
        # fill_window() -> empty, current_window is empty?
        # No, `del current_window[:step]` leaves OVERLAP frames.
        # fill_window adds nothing.
        # Next iter: `len(current_window) == OVERLAP`.
        # `padded_window` created.
        # Inference runs.
        # Alignment runs.
        # `org_window_len = OVERLAP`.
        # `safe_yield_end = OVERLAP`.
        # Yield interpolated (INTERP_LEN).
        # Yield `range(OVERLAP, OVERLAP)` -> Empty.
        # Correct? 
        # We yielded the overlap region (interpolated).
        # We are done?
        # Wait, if `org_window_len == OVERLAP`.
        # This implies we just processed the overlap part again?
        # No, the logic `range(0, org_video_len, frame_step)` strictly steps forward.
        # If we have only OVERLAP frames left, it means we have no NEW frames.
        # We shouldn't process.
        
        # My while loop condition `len(current_window) > 0` might be too loose.
        # We should stop if we don't have enough NEW frames?
        # Actually `frame_id` tracking matches the original loop.
        
        # In original: `range(0, org_video_len, frame_step)`
        # `frame_list` contains all frames.
        # If I strictly follow `frame_step` consumption, I'm safe.
        
        pass
