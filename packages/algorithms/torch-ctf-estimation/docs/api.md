# API reference

Everything below is importable from `torch_ctf_estimation` directly. `estimate_ctf` fits defocus (a 1D fit on the mean power spectrum, then 2D refinement across patches). `estimate_ctf_and_thickness` runs the same steps, also fits sample thickness, and returns a `CTFEstimationResult`. The pipeline stages are exported individually for building custom workflows.

## Estimating the CTF

::: torch_ctf_estimation.estimate_ctf.estimate_ctf
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf.estimate_ctf_and_thickness
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf.CTFEstimationResult
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Parameters

::: torch_ctf_estimation.models.input_models.OpticalParams
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.input_models.CTFFittingParams
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.input_models.ThicknessParams
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.input_models.LaserParams
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Results

::: torch_ctf_estimation.models.results_models.Defocus1DResults
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.results_models.Defocus2DResults
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.results_models.Thickness1DResults
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.models.results_models.Thickness2DResults
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Pipeline stages

::: torch_ctf_estimation.utils.prepare_image.prepare_image_for_ctf
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.utils.patches.extract_ctf_patches
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.utils.patches.compute_patch_power_spectra
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.utils.patches.normalised_patch_positions
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_1d.estimate_ctf_1d_utils.fit_background_spline_1d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_1d.estimate_ctf_1d.estimate_ctf_1d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils.estimate_defocus_2d_at_1x1
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d.estimate_ctf_2d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.utils.defocus_field_from_1d.defocus_field_from_1d_fits
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_1d.estimate_thickness_1d.estimate_thickness_1d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_2d.estimate_thickness_2d.estimate_thickness_2d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_1d.refine_defocus_and_thickness_1d.refine_defocus_and_thickness_1d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.estimate_ctf_2d.refine_defocus_and_thickness_2d.refine_defocus_and_thickness_2d
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Laser phase plate

::: torch_ctf_estimation.utils.laser_axis_mask.build_laser_axis_mask
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_ctf_estimation.utils.laser_axis_mask.apply_laser_axis_mask
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
