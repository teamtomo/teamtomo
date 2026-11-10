# API reference

Everything below is importable from `torch_tilt_series` directly. The loaders are
also available as class methods, `TiltSeries.from_aretomo_output(...)` and
`TiltSeries.from_etomo_directory(...)`. They need the optional IO dependencies:
`pip install torch-tilt-series[io]`.

## Loading

::: torch_tilt_series.io.from_aretomo_output
    options:
      show_root_heading: true
      heading_level: 3

::: torch_tilt_series.io.from_etomo_directory
    options:
      show_root_heading: true
      heading_level: 3

::: torch_tilt_series.io.load_tilt_series_images
    options:
      show_root_heading: true
      heading_level: 3

## Tilt series

::: torch_tilt_series.tilt_series.TiltSeries
    options:
      show_root_heading: true
      heading_level: 3

## Preprocessing

::: torch_tilt_series.preprocessing.preprocess_tilt_series_images
    options:
      show_root_heading: true
      heading_level: 3

::: torch_tilt_series.utils.normalize_on_central_crop
    options:
      show_root_heading: true
      heading_level: 3

::: torch_tilt_series.utils.subtract_plane
    options:
      show_root_heading: true
      heading_level: 3
