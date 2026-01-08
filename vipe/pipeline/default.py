# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import pickle

from pathlib import Path

import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    PrefetchVideoStream,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import VisualizationWriter, save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    AdaptiveDepthProcessor,
    FixedIntrinsicsProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        import json
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        # Use user-provided intrinsics if available, otherwise estimate with GeoCalib
        intrinsics_file = getattr(self.init_cfg, 'intrinsics_file', None)
        if intrinsics_file is not None:
            logger.info(f"Using fixed intrinsics from {intrinsics_file}")
            with open(intrinsics_file, 'r') as f:
                intr_data = json.load(f)
            init_processors.append(FixedIntrinsicsProcessor(
                fx=intr_data['fx'],
                fy=intr_data['fy'],
                cx=intr_data['cx'],
                cy=intr_data['cy'],
                camera_type=self.camera_type,
            ))
        else:
            init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def _save_pose_only(self, slam_output: SLAMOutput, artifact_path: io.ArtifactPath, view_idx: int, num_frames: int) -> None:
        """Save only poses and intrinsics from SLAM output, skipping all other artifacts.
        
        Output format: {base_path}/{artifact_name}/cameras.npz
        This matches the expected format for convert_vipe_cameras.py
        """
        import numpy as np
        
        # Output to {artifact_name}/cameras.npz format
        output_dir = artifact_path.base_path / artifact_path.artifact_name
        output_dir.mkdir(exist_ok=True, parents=True)
        pose_output_path = output_dir / "cameras.npz"
        
        # Get trajectory for this view
        if slam_output.rig is not None:
            trajectory = slam_output.get_view_trajectory(view_idx)
        else:
            trajectory = slam_output.trajectory
        
        # Convert SE3 trajectory to numpy arrays
        # trajectory is SE3 with shape (N,), we need c2w matrices
        pose_matrices = trajectory.matrix().cpu().numpy()  # (N, 4, 4)
        pose_inds = np.arange(len(pose_matrices))
        
        # Get intrinsics
        intrinsics = slam_output.intrinsics[view_idx].cpu().numpy()  # (4,) - fx, fy, cx, cy
        intrinsics_data = np.tile(intrinsics, (num_frames, 1))  # (N, 4)
        
        # Save poses and intrinsics together in cameras.npz
        np.savez(
            pose_output_path,
            data=pose_matrices,
            inds=pose_inds,
            intrinsics=intrinsics_data,
        )
        logger.info(f"Saved {len(pose_matrices)} poses to {pose_output_path}")

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        # [Optimized] Remove .cache() calls to prevent loading full video to memory
        # [Optimized] Remove .cache() calls to prevent loading full video to memory
        # [Stability] PrefetchVideoStream disabled due to FFmpeg threading issues.
        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream) for video_stream in video_streams
        ]

        # SLAM system consumes the streams first
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        # Pose-only mode: skip post-processing and artifact writing, save only poses
        if getattr(self.out_cfg, 'pose_only', False):
            logger.info("Pose-only mode enabled - skipping artifact writing")
            for view_idx, artifact_path in enumerate(artifact_paths):
                self._save_pose_only(slam_output, artifact_path, view_idx, len(video_streams[view_idx]))
            return annotate_output

        # Re-create streams for output pass (post-processing)
        # Note: Previous streams were consumed. We rely on underlying RawMp4Stream to support re-open.
        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Streaming Output Loop
        for i, (output_stream, artifact_path) in enumerate(zip(output_streams, artifact_paths)):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save SLAM map first if needed
            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

            writer = io.ArtifactWriter(artifact_path, output_stream.fps(), self.out_cfg.save_artifacts)
            
            viz_writer = None
            if self.out_cfg.save_viz:
                 viz_writer = VisualizationWriter(
                    artifact_path.meta_vis_path,
                    output_stream.fps(),
                    output_stream.frame_size(),
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                 )

            logger.info(f"Processing output stream {i} (streaming mode)...")
            
            # Save meta info
            if self.out_cfg.save_artifacts:
                 with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            from tqdm import tqdm
            for frame_idx, frame in tqdm(enumerate(output_stream), total=len(output_stream), desc="Writing artifacts"):
                writer.write(frame_idx, frame)
                if viz_writer:
                    viz_writer.write(frame_idx, frame)
                # Help GC
                del frame
            
            writer.close()
            if viz_writer:
                viz_writer.close()
                logger.info(f"Saved visualization to {artifact_path.meta_vis_path}")

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
