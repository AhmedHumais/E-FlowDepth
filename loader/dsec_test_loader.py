from __future__ import annotations

import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .utils.dsec_utils import RepresentationType, VoxelGrid, EventSlicer, flow_16bit_to_float
from .utils.loader_config import DSECTask, LoaderConfig


# ======================================================================================
# Unified sequence dataset
# ======================================================================================

class DSECSequenceTest(Dataset):
    def __init__(
        self,
        seq_path: Path,
        seq_name: str,
        sequence_index: int,
        config: LoaderConfig,
        disparity_timestamp_dir: Path,
        flow_timestamp_dir: Path,
    ) -> None:
        super().__init__()

        if not seq_path.is_dir():
            raise FileNotFoundError(f"Sequence path not found: {seq_path}")
        if config.representation_type != RepresentationType.VOXEL:
            raise NotImplementedError("Only RepresentationType.VOXEL is supported")

        self.seq_path = seq_path
        self.seq_name = seq_name
        self.sequence_index = sequence_index
        self.config = config
        self.task = config.normalized_task()

        self.height = 480
        self.width = 640
        self.num_bins = config.num_bins
        self.delta_t_us = config.delta_t_ms * 1000

        self.disparity_timestamp_dir = disparity_timestamp_dir
        self.flow_timestamp_dir = flow_timestamp_dir

        self.voxel_grid = VoxelGrid(
            (self.num_bins, self.height, self.width),
            normalize=True,
        )

        image_timestamps = np.loadtxt(self.seq_path / "image_timestamps.txt", dtype=np.int64)
        image_indices = np.arange(len(image_timestamps))
        self.timestamps = image_timestamps[::2][1:-1]
        self.indices = image_indices[::2][1:-1]

        self.flow_submission_indices = self._load_submission_indices(
            self.flow_timestamp_dir / f"{self.seq_name}.csv",
            column_index=2,
        )
        self.disparity_submission_indices = self._load_submission_indices(
            self.disparity_timestamp_dir / f"{self.seq_name}.csv",
            column_index=1,
        )

        self.h5_files: Dict[str, h5py.File] = {}
        self.event_slicers: Dict[str, EventSlicer] = {}
        self.rectify_maps: Dict[str, np.ndarray] = {}

        self._open_event_stream("left")
        if self.requires_disparity:
            self._open_event_stream("right")

        self._finalizer = weakref.finalize(self, self._close_all_h5, self.h5_files)

    @property
    def requires_disparity(self) -> bool:
        return self.task in {DSECTask.DISPARITY, DSECTask.BOTH}

    @staticmethod
    def _load_submission_indices(csv_path: Path, column_index: int) -> Optional[np.ndarray]:
        if not csv_path.is_file():
            return None

        data = np.genfromtxt(csv_path, delimiter=",")
        data = np.atleast_2d(data)
        return np.asarray(data[:, column_index], dtype=np.int64)

    def _open_event_stream(self, location: str) -> None:
        event_dir = self.seq_path / "events" / location
        events_file = event_dir / "events.h5"
        rectify_file = event_dir / "rectify_map.h5"

        if not events_file.is_file():
            raise FileNotFoundError(f"Missing event file: {events_file}")
        if not rectify_file.is_file():
            raise FileNotFoundError(f"Missing rectify map file: {rectify_file}")

        h5f = h5py.File(str(events_file), "r")
        self.h5_files[location] = h5f
        self.event_slicers[location] = EventSlicer(h5f)

        with h5py.File(str(rectify_file), "r") as h5_rect:
            self.rectify_maps[location] = h5_rect["rectify_map"][()]

    @staticmethod
    def _close_all_h5(h5_files: Dict[str, h5py.File]) -> None:
        for h5f in h5_files.values():
            try:
                h5f.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_image_width_height(self) -> Tuple[int, int]:
        return self.width, self.height

    def getHeightAndWidth(self) -> Tuple[int, int]:
        return self.height, self.width
    
    @staticmethod
    def get_disparity_map(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not filepath.is_file():
            raise FileNotFoundError(filepath)

        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        disp = disp_16bit.astype(np.float32) / 256.0
        valid = (disp > 0).astype(np.float32)
        return disp, valid

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str) -> np.ndarray:
        rectify_map = self.rectify_maps[location]

        if rectify_map.shape != (self.height, self.width, 2):
            raise ValueError(
                f"Unexpected rectify map shape for {location}: {rectify_map.shape}"
            )

        if x.size == 0:
            return np.empty((0, 2), dtype=rectify_map.dtype)

        if x.max() >= self.width or y.max() >= self.height:
            raise ValueError(f"Event coordinates exceed bounds for camera {location}")

        return rectify_map[y, x]

    def events_to_voxel_grid(self, p: np.ndarray, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        if t.size == 0:
            return torch.zeros((self.num_bins, self.height, self.width), dtype=torch.float32)

        t = (t - t[0]).astype(np.float32)
        if t[-1] > 0:
            t = t / t[-1]
        else:
            t = np.zeros_like(t, dtype=np.float32)

        event_data_torch = {
            "p": torch.from_numpy(p.astype(np.float32)),
            "t": torch.from_numpy(t),
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
        }
        return self.voxel_grid.convert(event_data_torch)

    def _build_event_volume(self, t_start_us: int, t_end_us: int, location: str) -> torch.Tensor:
        event_data = self.event_slicers[location].get_events(t_start_us, t_end_us)
        if event_data is None:
            raise RuntimeError(
                f"Could not retrieve events for sequence={self.seq_name}, "
                f"location={location}, window=({t_start_us}, {t_end_us})"
            )

        p = event_data["p"]
        t = event_data["t"]
        x = event_data["x"]
        y = event_data["y"]

        xy_rect = self.rectify_events(x, y, location)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        return self.events_to_voxel_grid(p, t, x_rect, y_rect)

    def _is_sequence_start(self, index: int) -> bool:
        if index == 0:
            return True
        diff = self.timestamps[index - 1] - self.timestamps[index]
        return diff > 101000

    def __getitem__(self, index: int) -> Dict[str, Union[int, bool, torch.Tensor]]:
        timestamp = int(self.timestamps[index])
        file_index = int(self.indices[index])
        
        output = {"file_index": file_index, 
                  "timestamp": timestamp, 
                  "sequence_index": self.sequence_index, 
                  "init_seq": self._is_sequence_start(index)}

        # Common left window [t - dt, t]
        output["event_volume_left_0"] = self._build_event_volume(timestamp - self.delta_t_us, timestamp, location="left")
        output["event_volume_left_1"] = self._build_event_volume(timestamp, timestamp + self.delta_t_us, location="left")
        output["save_flow_submission"] = self.flow_submission_indices is not None and file_index in self.flow_submission_indices

        # Disparity or Both: right window [t - dt, t]
        if self.requires_disparity:
            output["event_volume_right_0"] = self._build_event_volume(timestamp - self.delta_t_us, timestamp, location="right")
            output["save_disparity_submission"] = self.disparity_submission_indices is not None and file_index in self.disparity_submission_indices

        return output


# ======================================================================================
# Dataset provider
# ======================================================================================

class DSECTestDatasetProvider:
    """
    Directory structure expected:

        dataset_path/
            test/
                event_data/
                    seq_name_1/
                        events_left/
                            events.h5
                            rectify_map.h5
                        events_right/
                            events.h5
                            rectify_map.h5
                        image_timestamps.txt
                    seq_name_2/
                        ...
                test_disparity_timestamps/
                    seq_name_1.csv
                    seq_name_2.csv
                    ...
                test_forward_optical_flow_timestamps/
                    seq_name_1.csv
                    seq_name_2.csv
                    ...

    Notes:
    - dataset_root is always dataset_path / split
    - if task is flow or both, sequences without flow timestamp csv are omitted
    - if task is disparity only, missing disparity csv does not omit the sequence;
      save_disparity_submission will simply be False for all samples in that sequence
    """

    def __init__(
        self,
        dataset_path: Path,
        config: LoaderConfig,
    ) -> None:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        self.dataset_path = dataset_path
        self.config = config
        self.task = config.normalized_task()

        self.dataset_root = dataset_path / config.split
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Split path not found: {self.dataset_root}")

        self.event_data_dir = self.dataset_root / "event_data"
        self.disparity_timestamp_dir = self.dataset_root / "test_disparity_timestamps"
        self.flow_timestamp_dir = self.dataset_root / "test_forward_optical_flow_timestamps"

        if not self.event_data_dir.is_dir():
            raise FileNotFoundError(f"Missing event_data directory: {self.event_data_dir}")
        if not self.disparity_timestamp_dir.is_dir():
            raise FileNotFoundError(
                f"Missing disparity timestamp directory: {self.disparity_timestamp_dir}"
            )
        if not self.flow_timestamp_dir.is_dir():
            raise FileNotFoundError(
                f"Missing flow timestamp directory: {self.flow_timestamp_dir}"
            )

        self.sequence_names: List[str] = []
        self.datasets: List[DSECSequenceTest] = []

        self._build_datasets()
        self.dataset = ConcatDataset(self.datasets)

    def _flow_csv_exists(self, seq_name: str) -> bool:
        return (self.flow_timestamp_dir / f"{seq_name}.csv").is_file()

    def _build_datasets(self) -> None:
        sequence_dirs = sorted([p for p in self.event_data_dir.iterdir() if p.is_dir()])

        sequence_index = 0
        for seq_path in sequence_dirs:
            seq_name = seq_path.name

            # For flow or both, omit sequences that do not have a flow timestamp csv
            if self.task == DSECTask.FLOW and not self._flow_csv_exists(seq_name):
                # print(f"Warning: Skipping sequence '{seq_name}' because flow timestamp csv is missing")
                continue

            self.sequence_names.append(seq_name)
            self.datasets.append(
                DSECSequenceTest(
                    seq_path=seq_path,
                    seq_name=seq_name,
                    sequence_index=sequence_index,
                    config=self.config,
                    disparity_timestamp_dir=self.disparity_timestamp_dir,
                    flow_timestamp_dir=self.flow_timestamp_dir,
                )
            )
            sequence_index += 1

        if len(self.datasets) == 0:
            raise RuntimeError(
                f"No valid sequences found for task='{self.task.value}' in {self.event_data_dir}"
            )

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_name_mapping(self) -> List[str]:
        return self.sequence_names

    def summary(self, logger) -> None:
        logger.write_line(
            "================================== Dataloader Summary ====================================",
            True,
        )
        logger.write_line(f"Loader Type:\t\t{self.__class__.__name__}", True)
        logger.write_line(f"Split:\t\t\t{self.config.split}", True)
        logger.write_line(f"Task:\t\t\t{self.task.value}", True)
        logger.write_line(f"Number of Sequences:\t{len(self.sequence_names)}", True)
        logger.write_line(f"Number of Voxel Bins:\t{self.config.num_bins}", True)
        logger.write_line(f"Delta T (ms):\t\t{self.config.delta_t_ms}", True)