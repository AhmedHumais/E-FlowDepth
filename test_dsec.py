from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.dsec_test_loader import DSECTestDatasetProvider 
from loader.utils.loader_config import DSECTask, LoaderConfig
from model.eflowdisp import EFlowDisp
from utils import image_utils, visualizations
from loader.utils.dsec_utils import RepresentationType


# ======================================================================================
# Argument parsing
# ======================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSEC sequence inference and submission export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--task", type=str, default=None, required=True, choices=[task.value for task in DSECTask], help="Select from {flow, disparity, both}.",)
    parser.add_argument("--flow-iters", type=int, default=None, help="Override optical-flow refinement iterations.")
    parser.add_argument("--disp-iters", type=int, default=None, help="Override disparity refinement iterations.")

    return parser.parse_args()


# ======================================================================================
# Config helpers
# ======================================================================================

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return config


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_config(args.config)

    if args.task in [task.value for task in DSECTask]:
        config["task"] = args.task
    if args.flow_iters is not None:
        config["flow_iters"] = args.flow_iters
    if args.disp_iters is not None:
        config["disp_iters"] = args.disp_iters

    return normalize_and_validate_config(config)

def normalize_and_validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(config)

    required_keys = {"checkpoint", "dataset_path", "results_path", "split", "flow_iters", "disp_iters", "model",
                     "init_flow_iters", "init_disp_iters", "num_workers", "device", "save_visualizations", "strict_load"}

    missing = sorted(required_keys - set(normalized.keys()))
    if missing:
        raise ValueError(f"Missing required config entries: {missing}")

    for key in ("checkpoint", "dataset_path", "results_path"):
        normalized[key] = Path(normalized[key])

    if normalized["task"] not in {task.value for task in DSECTask}:
        raise ValueError(f"Invalid task: {normalized['task']}")

    for key in ("flow_iters", "disp_iters", "num_workers", "init_flow_iters", "init_disp_iters"):
        value = normalized[key]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"'{key}' must be a non-negative integer, got {value}")

    if not isinstance(normalized["split"], str) or len(normalized["split"]) == 0:
        raise ValueError("split must be a non-empty string")

    if not isinstance(normalized["device"], str) or len(normalized["device"]) == 0:
        raise ValueError("device must be a non-empty string")

    if not isinstance(normalized["save_visualizations"], bool):
        raise ValueError("save_visualizations must be a boolean")

    if not isinstance(normalized["strict_load"], bool):
        raise ValueError("strict_load must be a boolean")

    return normalized


# ======================================================================================
# Utilities
# ======================================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return value.item()
    return value


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(config: Dict, device: torch.device) -> torch.nn.Module:
    checkpoint_path = config["checkpoint"]
    strict = config["strict_load"]
    model_config = config["model"]
    model = EFlowDisp(config=model_config, n_first_channels=15)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    return model


# ======================================================================================
# Save helpers
# ======================================================================================

def save_flow_submission_png(flow: torch.Tensor, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    flow_np = flow.detach().cpu().numpy()  # [2, H, W]
    _, height, width = flow_np.shape

    flow_encoded = np.rint(flow_np * 128.0 + 2**15).astype(np.uint16).transpose(1, 2, 0)
    zero_channel = np.zeros((height, width, 1), dtype=np.uint16)
    png = np.concatenate((flow_encoded, zero_channel), axis=-1)

    imageio.imwrite(output_path, png, format="PNG-FI")


def save_disparity_submission_png(disparity: torch.Tensor, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    disparity_np = disparity.detach().cpu().numpy()
    if disparity_np.ndim == 3:
        disparity_np = disparity_np[0]

    png = np.array(disparity_np * 256.0, dtype=np.uint16)
    imageio.imwrite(output_path, png, format="PNG-FI")


def save_flow_visualization(flow: torch.Tensor, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    flow_np = flow.detach().cpu().numpy().transpose(1, 2, 0)
    vis = visualizations.flow_to_image(flow_np)
    imageio.imwrite(output_path, vis, format="PNG-FI")


def save_disparity_visualization(disparity: torch.Tensor, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    visualizations.visualize_disparity(disparity, display_on=False, save_path=str(output_path))


# ======================================================================================
# Inference runner
# ======================================================================================

class InferenceRunner:
    def __init__(self,model: torch.nn.Module, config: Dict[str, Any], sequence_names: List[str]) -> None:
        self.model = model
        self.device = resolve_device(config["device"])
        self.task = DSECTask(config["task"])

        self.flow_iters = config["flow_iters"]
        self.disp_iters = config["disp_iters"]
        self.init_flow_iters = config["init_flow_iters"]
        self.init_disp_iters = config["init_disp_iters"]

        self.results_path = config["results_path"]
        self.sequence_names = sequence_names
        self.save_visualizations = config["save_visualizations"]

        self.current_sequence_index: Optional[int] = None
        self.prev_flow_quarter: Optional[torch.Tensor] = None
        self.prev_disp_quarter: Optional[torch.Tensor] = None
        self.prev_flow_8th: Optional[torch.Tensor] = None

    def flow_submission_path(self, seq_name: str, file_index: int) -> Path:
        return self.results_path / "submission_flow" / seq_name / f"{file_index:06d}.png"

    def disparity_submission_path(self, seq_name: str, file_index: int) -> Path:
        return self.results_path / "submission_disparity" / seq_name / f"{file_index:06d}.png"

    def flow_vis_path(self, seq_name: str, file_index: int) -> Path:
        return self.results_path / "visualization_flow" / seq_name / f"{file_index:06d}.png"

    def disparity_vis_path(self, seq_name: str, file_index: int) -> Path:
        return self.results_path / "visualization_disparity" / seq_name / f"{file_index:06d}.png"

    def reset_state(self) -> None:
        self.prev_flow_quarter = None
        self.prev_disp_quarter = None
        self.prev_flow_8th = None

    def maybe_reset_state(self, batch: Dict[str, torch.Tensor]) -> None:
        sequence_index = int(scalar(batch["sequence_index"]))
        init_seq = bool(scalar(batch["init_seq"]))

        if self.current_sequence_index is None or sequence_index != self.current_sequence_index or init_seq:
            if self.current_sequence_index != sequence_index:
                print(f"Starting sequence: {self.sequence_names[sequence_index]}")
            else:
                print(f"Re-initializing sequence: {self.sequence_names[sequence_index]} (broken sequence at timestamp: {batch["timestamp"]})")
            self.reset_state()
            self.current_sequence_index = sequence_index

    def get_flow_prior(self) -> Optional[torch.Tensor]:
        if self.prev_flow_8th is not None:
            return image_utils.forward_interpolate_pytorch(self.prev_flow_8th)
        else:
            return None
        
    def get_disp_prior(self) -> Optional[torch.Tensor]:
        if self.prev_flow_quarter is not None and self.prev_disp_quarter is not None:
            return image_utils.forward_interp_disp(self.prev_disp_quarter, self.prev_flow_quarter)
        else:
            return None

    def run_flow_only(self, left_0: torch.Tensor, left_1: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            self.prev_flow_8th, flow_predictions = self.model(
                left_0,
                left_1,
                iters=self.flow_iters,
                flow_init=self.get_flow_prior(),
                recurr=True,
                flow_only=True,
            )

        final_flow = flow_predictions[-1]
        return final_flow

    def run_joint(self, left_0: torch.Tensor, left_1: torch.Tensor, right_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            self.prev_flow_8th, flow_predictions, disp_predictions = self.model(
                left_0,
                left_1,
                right_0,
                iters=self.init_flow_iters if self.prev_flow_8th is None else self.flow_iters,
                disp_iters=self.init_disp_iters if self.prev_disp_quarter is None else self.disp_iters,
                flow_init=self.get_flow_prior(),
                disp_init=self.get_disp_prior(),
                recurr=True,
            )
            final_flow = flow_predictions[-1]
            final_disp = disp_predictions[-1]

            self.prev_flow_quarter = F.interpolate(final_flow, scale_factor=(0.25, 0.25), mode="area") / 4.0
            self.prev_disp_quarter = F.interpolate(final_disp, scale_factor=(0.25, 0.25), mode="area") / 4.0


        return final_flow, final_disp

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        self.maybe_reset_state(batch)

        sequence_index = int(scalar(batch["sequence_index"]))
        file_index = int(scalar(batch["file_index"]))
        seq_name = self.sequence_names[sequence_index]

        left_0 = batch["event_volume_left_0"].to(self.device, dtype=torch.float32)
        left_1 = batch["event_volume_left_1"].to(self.device, dtype=torch.float32)

        if self.task == DSECTask.FLOW:
            final_flow = self.run_flow_only(left_0, left_1)

            if self.save_visualizations:
                save_flow_visualization(final_flow[0], self.flow_vis_path(seq_name, file_index))

            if bool(scalar(batch["save_flow_submission"])):
                save_flow_submission_png(final_flow[0], self.flow_submission_path(seq_name, file_index))
            return

        right_0 = batch["event_volume_right_0"].to(self.device, dtype=torch.float32)

        final_flow, final_disp = self.run_joint(left_0, left_1, right_0)

        if self.save_visualizations:
            if self.task == DSECTask.BOTH:
                save_flow_visualization(final_flow[0], self.flow_vis_path(seq_name, file_index))
            save_disparity_visualization(final_disp[0], self.disparity_vis_path(seq_name, file_index))

        if self.task == DSECTask.DISPARITY:
            if bool(scalar(batch["save_disparity_submission"])):
                save_disparity_submission_png(final_disp[0], self.disparity_submission_path(seq_name, file_index))
            return

        if bool(scalar(batch["save_flow_submission"])):
            save_flow_submission_png(final_flow[0], self.flow_submission_path(seq_name, file_index))

        if bool(scalar(batch["save_disparity_submission"])):
            save_disparity_submission_png(final_disp[0], self.disparity_submission_path(seq_name, file_index))


# ======================================================================================
# Main
# ======================================================================================


if __name__ == "__main__":
    args = parse_args()
    config = merge_config(args)

    device = resolve_device(config["device"])
    task = DSECTask(config["task"])

    ensure_dir(config["results_path"])
    
    loader_config = LoaderConfig(
        representation_type=RepresentationType.VOXEL,
        task=task,
        split=config["split"],
        delta_t_ms=100,
        num_bins=15,
    )

    provider = DSECTestDatasetProvider(dataset_path=config["dataset_path"], config=loader_config)

    dataset = provider.get_dataset()
    sequence_names = provider.get_name_mapping()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        # pin_memory=(device.type == "cuda"),
        pin_memory=False,
        drop_last=False,
    )

    model = build_model(config=config, device=device)

    runner = InferenceRunner(model=model, config=config, sequence_names=sequence_names)

    for batch in tqdm(loader, desc=f"Inference [{task.value}]"):
        runner.process_batch(batch)

    print(f"Finished. Results saved to: {config['results_path']}")

