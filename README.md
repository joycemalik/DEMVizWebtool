# DEM Visualiser Web Tool

An end-to-end web workflow for exploring high-resolution lunar Digital Elevation Models (DEMs) captured during the Chandrayaan-2 mission. The application couples centimetre-accurate geospatial cropping with an interactive WebGL renderer so scientists, designers, and educators can interrogate the Moon’s surface under custom lighting, export georeferenced snapshots, and trust that every height value is traceable back to the original DEM.

## Why this tool is one-of-a-kind

- **Survey-grade precision in a browser** – cropping is performed with sub-pixel accuracy using Rasterio’s affine math, so the geographic footprint shown in 3D exactly matches the coordinates you enter.
- **Mission-specific ergonomics** – defaults, CRS handling, and metadata panels are tuned for Chandrayaan-2 DEMs (Selenographic degrees), eliminating manual reprojection work.
- **Scientist-friendly transparency** – every crop exposes statistics, affine transforms, and CRS metadata so results are auditable and ready for downstream analysis.
- **Creative lighting sandbox** – three.js rendering exposes azimuth, elevation, and vertical exaggeration sliders, letting you stage photorealistic scenarios and export them as georeferenced GeoTIFF snapshots in one click.

## Feature Highlights

1. **Precise GeoTIFF cropping** – latitude/longitude bounds are validated, locked to source extents, and written with preserved CRS, resolution, nodata, and descriptive metadata.
2. **High-fidelity heightmaps** – large rasters are adaptively downsampled (bilinear resampling) to stay interactive while keeping elevation ranges intact for scientific use.
3. **Interactive 3D terrain viewer** – explore the DEM with orbit controls, adjustable sunlight, on-the-fly vertical exaggeration, and real-time normals for believable shading.
4. **Snapshot to GeoTIFF** – save any camera view as a 3-band GeoTIFF complete with synthetic geospatial metadata for archival or storytelling.
5. **Actionable metadata overlay** – see pixel dimensions, affine transform, geographic bounds, CRS, and elevation stats alongside the rendered scene.

## Technology Stack

- **Flask (Python)** – orchestrates uploads, validation, cropping, and API responses.
- **Rasterio** – performs geospatial math, windowed reads, resampling, and metadata preservation without OSGeo dependencies.
- **NumPy & Pillow** – array math, statistics, and snapshot image handling.
- **Three.js + Bootstrap** – cinematic WebGL terrain with a responsive UI layer.

## Getting Started

### Prerequisites

- Python 3.10+ (development tested on 3.13)
- A GeoTIFF DEM in geographic (degree) coordinates

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the application

```powershell
python "DEM Visualiser Web Tool/app.py"
```

Open the browser at <http://127.0.0.1:5000/> to upload a DEM, enter bounding coordinates, and launch the 3D viewer.

## Usage Workflow

1. **Upload & validate** – supply a GeoTIFF plus longitude/latitude bounds (upper-left and lower-right). Client-side and server-side checks prevent zero-area crops and out-of-bounds requests.
2. **Precision crop** – the backend converts your geographic window to exact pixel indices, crops using `rasterio.windows.from_bounds`, and records affine transforms, nodata, and elevation statistics.
3. **Inspect metadata** – the viewer displays pixel dimensions, CRS, bounding box, and min/mean/max elevations so you can confirm scientific validity.
4. **Explore in 3D** – orbit the terrain, dial in sun azimuth/elevation, and adjust vertical exaggeration (0.1×–20×) for storytelling or analysis.
5. **Export snapshots** – capture the current view as a GeoTIFF with bundled camera/sun metadata for reproducible renders.

## Accuracy & Data Fidelity

- **Affine-aware cropping** ensures the saved subset honours the original raster’s resolution and orientation down to the half-pixel.
- **Masked array handling** keeps nodata values from polluting statistics or height outputs. When required, nodata is filled with the local minimum so the mesh stays watertight without inventing new elevation.
- **Bilinear resampling** is only applied when the source window exceeds 2048 pixels in either dimension, striking a balance between interactivity and scientific accuracy.
- **Metadata-first design** exposes CRS, resolution, and transforms directly in the UI, making downstream GIS ingestion frictionless.

## Project Structure

```
.
├── DEM Visualiser Web Tool/
│   ├── app.py             # Flask application and geospatial processing pipeline
│   └── templates/         # Jinja templates (home, upload form, interactive viewer)
├── requirements.txt       # Python dependencies for rapid setup
└── .gitignore             # Repo hygiene (venv, caches, generated assets)
```

## Extending the Platform

- Swap in additional missions by adjusting CRS defaults or augmenting validation with mission-specific constraints.
- Add caching or tiling strategies (e.g., Cloud Optimised GeoTIFF serving) for ultra-large source datasets.
- Integrate mission metadata (illumination conditions, rover tracks) directly into the viewer using the provided metadata hooks.

## Credits & Acknowledgements

- Chandrayaan-2 DEM datasets curated by ISRO.
- Rasterio, Flask, and Three.js communities for the tooling that makes precise, beautiful lunar exploration accessible through a browser.



