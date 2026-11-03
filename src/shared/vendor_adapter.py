"""Thin imports/SMPL field compatibility for the unmodified pinned vendor."""
from __future__ import annotations
import importlib.util, json, logging, pickle, sys, threading
from pathlib import Path
from types import ModuleType
from typing import Any
import numpy as np
import torch
from config import PROJECT,IMUOPTIMIZE_COMMIT,PARENTS,fingerprint
from .storage import sha256_file
_IMPORT_LOCK=threading.RLock()
_VENDORS={}
_LOGGER=logging.getLogger("imuplace")
def verify_vendor(root: Path) -> None:
    manifest_path = root.parent / "imuoptimize.manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing vendor manifest: {manifest_path}. Follow README.md section 2 to download the pinned IMUOptimize source and generate third_party/imuoptimize.manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("commit") != IMUOPTIMIZE_COMMIT:
        raise RuntimeError("IMUOptimize commit differs from the audited pin. Follow README.md section 2 to restore the pinned IMUOptimize checkout and its checksum manifest; do not substitute another commit.")
    for relative, digest in manifest["files_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Vendored file missing or modified: {path}. Restore the unchanged pinned file using README.md section 2; put compatibility fixes in the adapter, not upstream files.")


def _load_module(name: str, path: Path, *, package: bool = False) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audited upstream module at {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class VendorAdapter:
    """Imports private upstream modules without importing upstream utils.py.

    Audited APIs: read_amass_npz(npz_path); transform_poses_trans(poses, trans);
    run_kinematics(model, poses, trans, betas); normalize_imu_data(acc, ori,
    num_joints); synthesize_imu_data(vert_model, vert, grot, requested_joints,
    smooth_n=4); ParametricModel.forward_kinematics(..., calc_mesh=True).
    """
    def __init__(self, root: Path, device: torch.device):
        self.root = root.resolve()
        self.device = device
        verify_vendor(self.root)
        prefix = "_imuplace_upstream_" + fingerprint([str(self.root), str(device)])[:12]
        art_name = prefix + ".articulate"
        namespace = ModuleType(prefix)
        namespace.__path__ = [str(self.root / "src")]
        sys.modules[prefix] = namespace
        art = ModuleType(art_name)
        art.__path__ = [str(self.root / "src" / "articulate")]
        art.__package__ = art_name
        sys.modules[art_name] = art
        art.math = _load_module(art_name + ".math", self.root / "src/articulate/math/__init__.py", package=True)
        model = _load_module(art_name + ".model", self.root / "src/articulate/model.py")
        art.ParametricModel = model.ParametricModel
        self.art = art
        shim = ModuleType(prefix + ".safe_utils")
        shim.torch_device = device
        from rich.console import Console
        shim.console = Console()
        shim.log_info = lambda message, webhook=False: _LOGGER.info("Upstream: %s", message)
        shim.log_warning = lambda message, webhook=False: _LOGGER.warning("Upstream: %s", message)
        shim.log_error = lambda message, webhook=False: _LOGGER.error("Upstream: %s", message)
        # Audited modules have absolute imports. Restore global names afterwards;
        # loaded modules retain private references to art and the local shim.
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in ("articulate", "utils")}
        try:
            sys.modules["articulate"] = art
            sys.modules["utils"] = shim
            self.pipeline = _load_module(prefix + ".pipeline", self.root / "src/data/synthesize/pipeline.py")
            self.raw_amass = _load_module(prefix + ".raw_amass", self.root / "src/data/load/raw_amass.py")
        finally:
            for name, original in previous.items():
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

    def model(self, path: Path, *, use_pose_blendshape: bool = False):
        """Populate the audited class from checked NumPy/SciPy SMPL fields.

        The upstream constructor assumes sparse J_regressor and exactly ten
        shapedirs. This adapter also accepts a dense regressor and explicitly
        selects the first ten shape components; FK itself is upstream code.
        """
        data = load_smpl_payload(path)
        body = self.art.ParametricModel.__new__(self.art.ParametricModel)
        field_map = {"_J_regressor": "J_regressor", "_skinning_weights": "weights",
                     "_posedirs": "posedirs", "_shapedirs": "shapedirs",
                     "_v_template": "v_template", "_J": "J"}
        for attribute, key in field_map.items():
            body.__dict__[attribute] = torch.from_numpy(data[key]).float().to(self.device)
        body.face = data["f"]
        body.parent = [None, *PARENTS[1:]]
        body.use_pose_blendshape = use_pose_blendshape
        return body


def vendor(device: str | torch.device = "cpu", root: Path | None = None) -> VendorAdapter:
    path = root or PROJECT / "third_party/imuoptimize"
    device = torch.device(device)
    key = (str(path.resolve()), str(device))
    with _IMPORT_LOCK:
        if key not in _VENDORS:
            _VENDORS[key] = VendorAdapter(path, device)
        return _VENDORS[key]


def load_smpl_payload(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing SMPL body model: {path}. Follow README.md section 3, SMPL setup.")
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                raw = {key: archive[key] for key in archive.files}
        else:
            with path.open("rb") as stream:
                raw = pickle.load(stream, encoding="latin1")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"{path} cannot load its legacy Python objects ({exc}). Use an official SMPL model serialized "
            "as NumPy arrays/SciPy sparse matrices, rather than Chumpy objects. Obtain a compatible "
            "model from the licensed download or export its numeric fields in an environment that "
            "already supports that legacy model. Required fields are listed in README.md section 3; this reader validates their shapes."
        ) from exc
    except (EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"Cannot decode SMPL model {path}: {exc}. Restore a complete trusted model file.") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: SMPL payload must be a dictionary.")
    required = ("J_regressor", "weights", "posedirs", "shapedirs", "v_template", "J", "f", "kintree_table")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{path}: missing SMPL fields {missing}. SMPL-X/SMPL+H files are not the required 24-joint body model.")
    data = {}
    for key in required:
        value = raw[key].toarray() if hasattr(raw[key], "toarray") else raw[key]
        try:
            data[key] = np.asarray(value)
        except Exception as exc:
            raise ValueError(f"{path}: cannot convert SMPL field {key} to a numeric array.") from exc
        if data[key].dtype.kind not in "fiu" or not np.isfinite(data[key]).all():
            raise ValueError(f"{path}: {key} must contain finite numeric values.")
    vertices = data["v_template"].shape[0]
    shapes = {"v_template": (vertices, 3), "J_regressor": (24, vertices),
              "weights": (vertices, 24), "J": (24, 3), "kintree_table": (2, 24),
              "posedirs": (vertices, 3, 207)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{path}: {key} has shape {data[key].shape}; required {shape}.")
    shapedirs = data["shapedirs"]
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != (vertices, 3) or shapedirs.shape[2] < 10:
        raise ValueError(f"{path}: shapedirs must be [V,3,B] with B>=10; received {shapedirs.shape}.")
    if data["f"].ndim != 2 or data["f"].shape[1] != 3:
        raise ValueError(f"{path}: faces must have shape [F,3].")
    if not np.equal(data["f"], np.floor(data["f"])).all():
        raise ValueError(f"{path}: face indices must be integers.")
    if not np.array_equal(data["kintree_table"][1], np.arange(24)):
        raise ValueError(f"{path}: unsupported nonstandard SMPL joint identifiers.")
    if not np.array_equal(data["kintree_table"][0, 1:], np.asarray(PARENTS[1:])):
        raise ValueError(f"{path}: SMPL parent tree differs from the audited 24-joint convention.")
    if np.any(data["f"] < 0) or np.any(data["f"] >= vertices):
        raise ValueError(f"{path}: face indices are outside the vertex array.")
    data["shapedirs"] = np.ascontiguousarray(shapedirs[..., :10], dtype=np.float32)
    for key in ("v_template", "J_regressor", "weights", "J", "posedirs"):
        data[key] = np.ascontiguousarray(data[key], dtype=np.float32)
    data["f"] = np.ascontiguousarray(data["f"], dtype=np.int64)
    return data
