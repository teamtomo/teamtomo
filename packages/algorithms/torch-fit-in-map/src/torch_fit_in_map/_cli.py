"""Command-line interfaces for torch-fit-in-map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

# Extensions treated as atomic-model inputs
_PDB_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent"}
# Extensions treated as MRC density maps
_MRC_SUFFIXES = {".mrc", ".map", ".rec"}

align_app = typer.Typer(
    name="torch-fit-in-map",
    help="Rigid-body volume alignment for cryo-EM density maps.",
    add_completion=False,
)

simulate_app = typer.Typer(
    name="torch-simulate-density",
    help="Simulate an MRC density map from an atomic model.",
    add_completion=False,
)

fit_in_atomic_model_app = typer.Typer(
    name="torch-fit-in-atomic-model",
    help="Fit an experimental density map into the coordinate frame of an atomic model.",
    add_completion=False,
)


def _result_to_dict(result: object) -> dict[str, object]:
    """Serialise an AlignmentResult to a JSON-safe dict."""
    from ._result import AlignmentResult

    r: AlignmentResult = result  # type: ignore[assignment]
    out: dict[str, object] = {
        "score": r.score,
        "rotation_matrix_zyx": r.rotation_matrix.tolist(),
        "translation_pixels_zyx": r.translation_pixels.tolist(),
    }
    if r.translation_angstroms is not None:
        out["translation_angstroms_zyx"] = r.translation_angstroms.tolist()
    return out


@simulate_app.command()
def simulate(
    model: Annotated[Path, typer.Argument(help="Atomic model (.pdb/.cif/.mmcif).")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output MRC path.")],
    pixel_size: Annotated[
        float, typer.Option("--pixel-size", help="Pixel size in Å.")
    ],
    box_size: Annotated[
        int, typer.Option("--box-size", help="Cubic box size in voxels.")
    ],
    desired_resolution: Annotated[
        Optional[float],
        typer.Option(
            "--desired-resolution",
            help=(
                "Low-pass filter the output to this resolution in Å. "
                "Must be >= 2 × pixel_size (Nyquist limit)."
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device(s), e.g. 'cpu', 'cuda', 'all', or '0,1' for multi-GPU.",
        ),
    ] = "auto",
) -> None:
    """Simulate an MRC density map from an atomic model.

    Uses the configured DensitySimulator (requires torch-calculate-electrostatic-potential
    once available).
    """
    import torch

    from ._io import _save_mrc
    from ._simulate import DEFAULT_SIMULATOR

    if desired_resolution is not None and desired_resolution < 2.0 * pixel_size:
        typer.echo(
            f"Error: --desired-resolution ({desired_resolution} Å) must be >= "
            f"2 × pixel_size ({2.0 * pixel_size} Å).",
            err=True,
        )
        raise typer.Exit(code=1)

    # Device parsing
    if device == "auto":
        devs = ["cuda" if torch.cuda.is_available() else "cpu"]
    elif device == "all":
        n = torch.cuda.device_count()
        devs = [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    elif "," in device:
        devs = [
            f"cuda:{d.strip()}" if d.strip().isdigit() else d.strip()
            for d in device.split(",")
        ]
    else:
        devs = [device]

    primary_device = torch.device(devs[0])

    if output.suffix.lower() not in _MRC_SUFFIXES:
        typer.echo(
            f"Error: Incompatible output extension '{output.suffix}' for simulated density. "
            f"Expected an MRC extension like {', '.join(_MRC_SUFFIXES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    from ._io import _pdb_centroid_xyz

    typer.echo(
        f"Simulating density from {model} at {pixel_size} Å/px, box {box_size}³ ...",
        err=True,
    )
    density = DEFAULT_SIMULATOR.simulate(
        pdb_path=model,
        pixel_size=pixel_size,
        box_size=box_size,
        device=primary_device,
    )

    if desired_resolution is not None:
        from torch_fourier_filter.bandpass import low_pass_filter

        cutoff = pixel_size / desired_resolution  # normalised frequency (0–0.5)
        lp = low_pass_filter(
            cutoff=cutoff,
            falloff=0.02,
            image_shape=density.shape,  # type: ignore[arg-type]
            rfft=True,
            fftshift=False,
            device=primary_device,
        )
        ft = torch.fft.rfftn(density, norm="ortho")
        density = torch.fft.irfftn(ft * lp, s=density.shape, norm="ortho")
        typer.echo(
            f"Low-pass filtered to {desired_resolution} Å (cutoff={cutoff:.3f}).",
            err=True,
        )

    # Set MRC origin so the simulated map co-localises with the PDB in ChimeraX.
    # Simulation centres atoms at box_centre_A; the origin is where voxel [0,0,0] sits.
    centroid_xyz = _pdb_centroid_xyz(model)
    box_centre_a = (box_size - 1) / 2.0 * pixel_size
    origin_xyz = (
        centroid_xyz[0] - box_centre_a,
        centroid_xyz[1] - box_centre_a,
        centroid_xyz[2] - box_centre_a,
    )
    _save_mrc(output, density, pixel_size=pixel_size, origin_xyz=origin_xyz)
    typer.echo(f"Saved simulated density to {output}", err=True)


@align_app.command()
def align(
    reference: Annotated[Path, typer.Argument(help="Reference MRC density map.")],
    mobile: Annotated[
        Path,
        typer.Argument(
            help="Mobile file to align: MRC map or atomic model (.pdb/.cif/.mmcif)."
        ),
    ],
    angular_step: Annotated[
        float, typer.Option("--angular-step", help="Angular search step in degrees.")
    ] = 15.0,
    symmetry: Annotated[
        str,
        typer.Option(
            "--symmetry",
            help=(
                "Point-group symmetry of the reference map, e.g. C1, C4, D2, T, O, I. "
                "Restricts the SO(3) search to the asymmetric unit (speeds up search)."
            ),
        ),
    ] = "C1",
    n_iter: Annotated[
        int,
        typer.Option("--n-iter", help="Gradient refinement iterations (0 = skip)."),
    ] = 100,
    optimizer: Annotated[
        str,
        typer.Option("--optimizer", help="Refinement optimizer ('lbfgs' or 'adam')."),
    ] = "lbfgs",
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help="Refinement learning rate (defaults: lbfgs=1.0, adam=0.01)."),
    ] = None,
    n_start: Annotated[
        int,
        typer.Option(
            "--n-start",
            help=(
                "Number of top exhaustive-search poses to refine independently. "
                "The best result is returned.  Higher values improve robustness."
            ),
        ),
    ] = 1,
    mask: Annotated[
        Optional[Path], typer.Option("--mask", help="Optional MRC soft-mask.")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            help="Save the aligned result: MRC volume (map input) or PDB/CIF atomic model (PDB/CIF input).",
        ),
    ] = None,
    pixel_size: Annotated[
        Optional[float],
        typer.Option(
            "--pixel-size",
            help=(
                "Pixel size in Å for PDB simulation. "
                "Defaults to the pixel size read from the REFERENCE MRC header."
            ),
        ),
    ] = None,
    box_size: Annotated[
        Optional[int],
        typer.Option(
            "--box-size",
            help=(
                "Box size in voxels for PDB simulation. "
                "Defaults to the largest dimension of the REFERENCE map."
            ),
        ),
    ] = None,
    desired_resolution: Annotated[
        Optional[float],
        typer.Option(
            "--desired-resolution",
            help=(
                "Low-pass filter the simulated density to this resolution in Å "
                "before alignment (only used when MOBILE is a PDB). "
                "Must be >= 2 × pixel_size (Nyquist limit)."
            ),
        ),
    ] = None,
    save_simulated: Annotated[
        Optional[Path],
        typer.Option(
            "--save-simulated",
            help="Save simulated density MRC here (only used when MOBILE is a PDB).",
        ),
    ] = None,
    output_json: Annotated[
        Optional[Path],
        typer.Option("--output-json", help="Write result JSON to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet/--no-quiet",
            "-q/-Q",
            help="Suppress progress bars and stdout output.  Requires --output or --output-json.",
        ),
    ] = False,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device(s), e.g. 'cpu', 'cuda', 'all', or '0,1' for multi-GPU.",
        ),
    ] = "auto",
) -> None:
    """Align MOBILE onto REFERENCE.

    MOBILE can be an MRC density map or an atomic model (.pdb / .cif / .mmcif).
    The mode is detected automatically from the file extension.
    """
    import torch

    from ._config import ExhaustiveSearchConfig, GradientRefinementConfig
    from ._io import (
        _load_mrc,
        _read_mrc_header,
        _save_mrc,
        fit_pdb_in_map_from_files,
        fit_map_in_map_from_files,
        transform_atomic_model,
    )
    from . import apply_alignment as _apply_alignment

    if quiet and output is None and output_json is None:
        typer.echo(
            "Error: --quiet requires --output or --output-json (otherwise all results are lost).",
            err=True,
        )
        raise typer.Exit(code=1)

    # Device parsing
    if device == "auto":
        devs = ["cuda" if torch.cuda.is_available() else "cpu"]
    elif device == "all":
        n = torch.cuda.device_count()
        devs = [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    elif "," in device:
        devs = [
            f"cuda:{d.strip()}" if d.strip().isdigit() else d.strip()
            for d in device.split(",")
        ]
    else:
        devs = [device]

    primary_device = torch.device(devs[0])

    exhaustive_cfg = ExhaustiveSearchConfig(
        angular_step_degrees=angular_step,
        symmetry=symmetry,
        n_start=n_start,
        devices=devs,
    )

    is_pdb = mobile.suffix.lower() in _PDB_SUFFIXES

    if output is not None:
        out_suffix = output.suffix.lower()
        if is_pdb and out_suffix not in _PDB_SUFFIXES:
            typer.echo(
                f"Error: Incompatible output extension '{out_suffix}' for atomic model input. "
                f"Expected one of: {', '.join(_PDB_SUFFIXES)}",
                err=True,
            )
            raise typer.Exit(code=1)
        if not is_pdb and out_suffix not in _MRC_SUFFIXES:
            # We are more lenient here but we should at least warn or error if it's a PDB suffix
            if out_suffix in _PDB_SUFFIXES:
                typer.echo(
                    f"Error: Incompatible output extension '{out_suffix}' for MRC map input. "
                    f"Expected an MRC extension like {', '.join(_MRC_SUFFIXES)}",
                    err=True,
                )
                raise typer.Exit(code=1)

    if save_simulated is not None and save_simulated.suffix.lower() not in _MRC_SUFFIXES:
        typer.echo(
            f"Error: Incompatible output extension '{save_simulated.suffix}' for --save-simulated. "
            f"Expected an MRC extension like {', '.join(_MRC_SUFFIXES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if n_iter > 0:
        grad_kwargs = {
            "optimizer": optimizer,
            "n_iterations": n_iter,
            "devices": devs,
        }
        if learning_rate is not None:
            grad_kwargs["learning_rate"] = learning_rate
        gradient_cfg = GradientRefinementConfig(**grad_kwargs)  # type: ignore[arg-type]
    else:
        gradient_cfg = None

    is_pdb = mobile.suffix.lower() in _PDB_SUFFIXES

    if is_pdb:
        result = fit_pdb_in_map_from_files(
            mobile_pdb_path=mobile,
            reference_map_path=reference,
            pixel_size_angstroms=pixel_size,
            box_size=box_size,
            desired_resolution_angstroms=desired_resolution,
            save_simulated=save_simulated is not None,
            simulated_output_path=None,
            exhaustive_config=exhaustive_cfg,
            gradient_config=gradient_cfg,
            mask_path=mask,
            device=primary_device,
            verbose=not quiet,
        )
        mobile_tensor: torch.Tensor | None = (
            result.simulated_volume if result.simulated_volume is not None else None
        )
        mobile_pixel_size = pixel_size

        if save_simulated is not None and result.simulated_volume is not None:
            ref_shape, ref_px, ref_origin = _read_mrc_header(reference)
            aligned_sim = _apply_alignment(result.simulated_volume.to(primary_device), result)
            _save_mrc(save_simulated, aligned_sim, pixel_size=ref_px, origin_xyz=ref_origin)
            typer.echo(f"Aligned simulated density saved to {save_simulated}", err=True)
    else:
        result = fit_map_in_map_from_files(
            mobile_path=mobile,
            reference_path=reference,
            exhaustive_config=exhaustive_cfg,
            gradient_config=gradient_cfg,
            mask_path=mask,
            device=primary_device,
            verbose=not quiet,
        )
        mobile_tensor = None
        mobile_pixel_size = None

    data = _result_to_dict(result)
    if not quiet:
        typer.echo(json.dumps(data, indent=2))

    if output_json is not None:
        output_json.write_text(json.dumps(data, indent=2))
        typer.echo(f"Result written to {output_json}", err=True)

    if output is not None:
        if is_pdb:
            ref_shape, ref_px, ref_origin = _read_mrc_header(reference)
            sim_box_size = box_size if box_size is not None else max(ref_shape)
            transform_atomic_model(
                input_path=mobile,
                output_path=output,
                rotation_matrix_zyx=result.rotation_matrix,
                translation_pixels_zyx=result.translation_pixels,
                pixel_size=ref_px,
                box_shape=ref_shape,
                sim_box_size=sim_box_size,
                ref_origin_xyz=ref_origin,
            )
            typer.echo(f"Transformed atomic model saved to {output}", err=True)
        else:
            if mobile_tensor is None:
                mobile_tensor, mob_px = _load_mrc(mobile)
                mobile_pixel_size = mob_px
            aligned = _apply_alignment(mobile_tensor.to(primary_device), result)
            _save_mrc(output, aligned, pixel_size=mobile_pixel_size or 1.0)
            typer.echo(f"Aligned volume saved to {output}", err=True)


@fit_in_atomic_model_app.command()
def fit_in_atomic_model(
    reference: Annotated[
        Path,
        typer.Argument(help="Reference atomic model (.pdb / .cif / .mmcif)."),
    ],
    mobile: Annotated[
        Path,
        typer.Argument(help="Mobile MRC density map to fit into the atomic model frame."),
    ],
    angular_step: Annotated[
        float, typer.Option("--angular-step", help="Angular search step in degrees.")
    ] = 15.0,
    symmetry: Annotated[
        str,
        typer.Option(
            "--symmetry",
            help=(
                "Point-group symmetry of the reference model, e.g. C1, C4, D2. "
                "Restricts the SO(3) search to the asymmetric unit."
            ),
        ),
    ] = "C1",
    n_iter: Annotated[
        int,
        typer.Option("--n-iter", help="Gradient refinement iterations (0 = skip)."),
    ] = 100,
    optimizer: Annotated[
        str,
        typer.Option("--optimizer", help="Refinement optimizer ('lbfgs' or 'adam')."),
    ] = "lbfgs",
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help="Refinement learning rate (defaults: lbfgs=1.0, adam=0.01)."),
    ] = None,
    n_start: Annotated[
        int,
        typer.Option(
            "--n-start",
            help=(
                "Number of top exhaustive-search poses to refine independently. "
                "The best result is returned."
            ),
        ),
    ] = 1,
    mask: Annotated[
        Optional[Path], typer.Option("--mask", help="Optional MRC soft-mask.")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            help="Save the aligned density map as MRC.",
        ),
    ] = None,
    pixel_size: Annotated[
        Optional[float],
        typer.Option(
            "--pixel-size",
            help=(
                "Pixel size in Å for PDB simulation. "
                "Defaults to the pixel size read from the mobile MRC header."
            ),
        ),
    ] = None,
    box_size: Annotated[
        Optional[int],
        typer.Option(
            "--box-size",
            help=(
                "Box size in voxels for PDB simulation. "
                "Defaults to the largest dimension of the mobile map."
            ),
        ),
    ] = None,
    desired_resolution: Annotated[
        Optional[float],
        typer.Option(
            "--desired-resolution",
            help=(
                "Low-pass filter the simulated reference density to this resolution in Å "
                "before alignment.  Must be >= 2 × pixel_size."
            ),
        ),
    ] = None,
    save_simulated: Annotated[
        Optional[Path],
        typer.Option("--save-simulated", help="Save simulated reference density MRC here."),
    ] = None,
    output_json: Annotated[
        Optional[Path],
        typer.Option("--output-json", help="Write result JSON to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet/--no-quiet",
            "-q/-Q",
            help="Suppress progress output.  Requires --output or --output-json.",
        ),
    ] = False,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device(s), e.g. 'cpu', 'cuda', 'all', or '0,1' for multi-GPU.",
        ),
    ] = "auto",
) -> None:
    """Fit MOBILE (density map) into the coordinate frame of REFERENCE (atomic model).

    The atomic model is simulated as a density map and used as the reference;
    the experimental density map is the mobile.  For the inverse (fit a PDB
    into a density map), use ``torch-fit-in-map``.
    """
    import torch

    from ._config import ExhaustiveSearchConfig, GradientRefinementConfig
    from ._io import (
        _load_mrc,
        _save_mrc,
        fit_map_in_pdb_from_files,
    )
    from . import apply_alignment as _apply_alignment

    if quiet and output is None and output_json is None:
        typer.echo(
            "Error: --quiet requires --output or --output-json (otherwise all results are lost).",
            err=True,
        )
        raise typer.Exit(code=1)

    if reference.suffix.lower() not in _PDB_SUFFIXES:
        typer.echo(
            f"Error: REFERENCE must be an atomic model "
            f"({', '.join(_PDB_SUFFIXES)}), got '{reference.suffix}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if mobile.suffix.lower() not in _MRC_SUFFIXES:
        typer.echo(
            f"Error: MOBILE must be an MRC density map "
            f"({', '.join(_MRC_SUFFIXES)}), got '{mobile.suffix}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if output is not None and output.suffix.lower() not in _MRC_SUFFIXES:
        typer.echo(
            f"Error: --output must have an MRC extension ({', '.join(_MRC_SUFFIXES)}), "
            f"got '{output.suffix}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if save_simulated is not None and save_simulated.suffix.lower() not in _MRC_SUFFIXES:
        typer.echo(
            f"Error: --save-simulated must have an MRC extension ({', '.join(_MRC_SUFFIXES)}), "
            f"got '{save_simulated.suffix}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Device parsing
    if device == "auto":
        devs = ["cuda" if torch.cuda.is_available() else "cpu"]
    elif device == "all":
        n = torch.cuda.device_count()
        devs = [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    elif "," in device:
        devs = [
            f"cuda:{d.strip()}" if d.strip().isdigit() else d.strip()
            for d in device.split(",")
        ]
    else:
        devs = [device]

    primary_device = torch.device(devs[0])

    exhaustive_cfg = ExhaustiveSearchConfig(
        angular_step_degrees=angular_step,
        symmetry=symmetry,
        n_start=n_start,
        devices=devs,
    )

    if n_iter > 0:
        grad_kwargs: dict[str, object] = {
            "optimizer": optimizer,
            "n_iterations": n_iter,
            "devices": devs,
        }
        if learning_rate is not None:
            grad_kwargs["learning_rate"] = learning_rate
        gradient_cfg = GradientRefinementConfig(**grad_kwargs)  # type: ignore[arg-type]
    else:
        gradient_cfg = None

    result = fit_map_in_pdb_from_files(
        mobile_map_path=mobile,
        reference_pdb_path=reference,
        pixel_size_angstroms=pixel_size,
        box_size=box_size,
        desired_resolution_angstroms=desired_resolution,
        save_simulated=save_simulated is not None,
        exhaustive_config=exhaustive_cfg,
        gradient_config=gradient_cfg,
        mask_path=mask,
        device=primary_device,
        verbose=not quiet,
    )

    if save_simulated is not None and result.simulated_volume is not None:
        mob_map, mob_px = _load_mrc(mobile)
        _save_mrc(
            save_simulated,
            result.simulated_volume,
            pixel_size=pixel_size or mob_px,
        )
        typer.echo(f"Simulated reference density saved to {save_simulated}", err=True)

    data = _result_to_dict(result)
    if not quiet:
        typer.echo(json.dumps(data, indent=2))

    if output_json is not None:
        output_json.write_text(json.dumps(data, indent=2))
        typer.echo(f"Result written to {output_json}", err=True)

    if output is not None:
        mob_map, mob_px = _load_mrc(mobile)
        aligned = _apply_alignment(mob_map.to(primary_device), result)
        _save_mrc(output, aligned, pixel_size=mob_px)
        typer.echo(f"Aligned density map saved to {output}", err=True)
