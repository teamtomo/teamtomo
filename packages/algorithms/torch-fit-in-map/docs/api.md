# API reference

Everything below is importable from `torch_fit_in_map` directly. `fit_structure_in_map`, `fit_map_in_map` and `fit_map_in_structure` align a mobile structure or map to a reference and return an `AlignmentResult`; `apply_alignment` and `apply_alignment_to_structure` apply it. The search stages and density simulators they use are exported for finer control.

## Fitting

::: torch_fit_in_map.fit_structure_in_map
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map.fit_map_in_map
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map.fit_map_in_structure
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._result.AlignmentResult
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map.apply_alignment
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._atoms.apply_alignment_to_structure
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Search stages

::: torch_fit_in_map._exhaustive.exhaustive_search
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._gradient.gradient_refine
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._projection.projection_align
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._config.ExhaustiveSearchConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._config.GradientRefinementConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._config.ProjectionAlignmentConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Simulating density

::: torch_fit_in_map._simulate.DensitySimulator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._simulate.PotentialSimulator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._config.PotentialSimulatorConfig
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._simulate.DEFAULT_SIMULATOR
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._simulate.DEFAULT_POTENTIAL_SIMULATOR
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Preprocessing

::: torch_fit_in_map._preprocess.normalise_voxel_sizes
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: torch_fit_in_map._preprocess.crop_or_pad_to_shape
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
