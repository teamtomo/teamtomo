# API reference

Everything below is importable from `torch_motion_correction` directly. Motion is estimated as a `DeformationField`, with `estimate_global_motion` (cross-correlation) or `estimate_local_motion` (gradient-based optimisation), and then applied to the movie with `correct_motion`.

## Estimating motion

::: torch_motion_correction.estimate_motion_xc.estimate_global_motion
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.estimate_motion_optimizer.estimate_local_motion
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.estimate_motion_xc.estimate_motion_cross_correlation_patches
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.deformation_field.DeformationField
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Correcting motion

::: torch_motion_correction.correct_motion.correct_motion
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.correct_motion.correct_motion_fast
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.correct_motion.correct_motion_slow
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.correct_motion.correct_motion_two_grids
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.correct_motion.get_pixel_shifts
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Configuration

::: torch_motion_correction.types.PatchSamplingConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.types.FourierFilterConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.types.OptimizationConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.types.XCRefinementConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Tracking optimisation

::: torch_motion_correction.optimization_state.OptimizationState
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_motion_correction.optimization_state.OptimizationTracker
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
