from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import rasterio
import rasterio.enums
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.coords import BoundingBox
from typing import Any, Dict
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CROPPED_FOLDER'] = 'cropped'
app.config['SNAPSHOTS_FOLDER'] = 'snapshots'

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPPED_FOLDER'], exist_ok=True)
os.makedirs(app.config['SNAPSHOTS_FOLDER'], exist_ok=True)


def parse_coordinate_form(form: Any) -> Dict[str, float]:
    """Parse and validate coordinate fields sent from the UI."""
    required_fields = ('ulx_geo', 'uly_geo', 'lrx_geo', 'lry_geo')
    missing = [field for field in required_fields if not form.get(field)]
    if missing:
        raise ValueError(f"Missing coordinate value for: {', '.join(missing)}")

    try:
        ulx = float(str(form['ulx_geo']).strip())
        uly = float(str(form['uly_geo']).strip())
        lrx = float(str(form['lrx_geo']).strip())
        lry = float(str(form['lry_geo']).strip())
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("All coordinates must be valid floating point numbers.") from exc

    if ulx == lrx or uly == lry:
        raise ValueError("Coordinates describe a zero-area region. Please provide distinct upper-left and lower-right points.")

    # For geographic coordinates (lat increases northward) ensure ordering
    if uly < lry:
        raise ValueError("Upper-left latitude must be greater than lower-right latitude (north to south).")

    return {
        'ulx': ulx,
        'uly': uly,
        'lrx': lrx,
        'lry': lry,
    }


def build_elevation_stats(array: np.ma.MaskedArray) -> Dict[str, float]:
    """Return min/mean/max statistics for a masked elevation array."""
    if array.count() == 0:
        return {'min': float('nan'), 'max': float('nan'), 'mean': float('nan')}
    return {
        'min': float(array.min()),
        'max': float(array.max()),
        'mean': float(array.mean()),
    }


def get_raster_metadata(file_path: str, include_preview: bool = False, preview_max: int = 512) -> Dict[str, Any]:
    """Extract raster metadata and optionally include a downsampled preview for UI selection."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        transform = src.transform
        resolution = (abs(transform.a), abs(transform.e))
        metadata: Dict[str, Any] = {
            'file_path': file_path,
            'width': int(src.width),
            'height': int(src.height),
            'bounds': {
                'left': float(bounds.left),
                'right': float(bounds.right),
                'top': float(bounds.top),
                'bottom': float(bounds.bottom),
            },
            'crs': str(src.crs) if src.crs else None,
            'resolution': [float(resolution[0]), float(resolution[1])],
            'transform': list(transform)[:6] if transform else None,
            'bands': int(src.count),
            'dtype': src.dtypes[0] if src.count > 0 else None,
        }

        if include_preview and src.count > 0:
            try:
                preview_scale = max(src.width, src.height) / preview_max if max(src.width, src.height) > preview_max else 1.0
                preview_width = max(1, int(src.width / preview_scale))
                preview_height = max(1, int(src.height / preview_scale))

                preview_array = src.read(
                    1,
                    out_shape=(preview_height, preview_width),
                    resampling=rasterio.enums.Resampling.bilinear,
                    masked=True,
                )

                if preview_array.dtype.kind not in {'f'}:
                    preview_array = preview_array.astype('float32')

                preview_filled = preview_array.filled(np.nan)
                valid_mask = ~np.isnan(preview_filled)

                if valid_mask.any():
                    low = float(np.nanpercentile(preview_filled, 2))
                    high = float(np.nanpercentile(preview_filled, 98))
                    if np.isclose(low, high):
                        low, high = float(np.nanmin(preview_filled)), float(np.nanmax(preview_filled))
                    if np.isclose(low, high):
                        scaled = np.zeros_like(preview_filled, dtype=np.uint8)
                    else:
                        scaled = np.clip((preview_filled - low) / (high - low), 0, 1)
                        scaled = (scaled * 255).astype(np.uint8)
                else:
                    scaled = np.zeros((preview_height, preview_width), dtype=np.uint8)

                image = Image.fromarray(scaled, mode='L')
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                preview_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

                metadata['preview'] = {
                    'image': f'data:image/png;base64,{preview_b64}',
                    'width': preview_width,
                    'height': preview_height,
                }
            except Exception as exc:  # pragma: no cover - defensive preview generation guard
                print(f"Preview generation failed: {exc}")
                metadata['preview'] = None
                metadata['preview_error'] = str(exc)

        return metadata

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html', bounds_metadata=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return render_template('crop.html', file_path=file.filename)
    return 'File not uploaded', 400


@app.route('/bounds', methods=['POST'])
def raster_bounds():
    if 'tiff-file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['tiff-file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.tif', '.tiff')):
        return jsonify({'error': 'Only GeoTIFF (.tif/.tiff) files are supported.'}), 400

    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)

    try:
        metadata = get_raster_metadata(saved_path, include_preview=True)
        return jsonify({'metadata': metadata})
    except Exception as exc:
        print(f"Error extracting bounds: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Unable to read raster bounds. Please verify the file.'}), 500

@app.route('/crop', methods=['POST'])
def crop_file():
    input_file = os.path.join(app.config['UPLOAD_FOLDER'], request.form['file_path'])
    output_file = os.path.join(app.config['CROPPED_FOLDER'], 'cropped_' + request.form['file_path'])
    bounds_metadata = None
    if os.path.exists(input_file):
        try:
            bounds_metadata = get_raster_metadata(input_file, include_preview=True)
        except Exception as exc:
            print(f"Warning: unable to extract metadata for crop_file: {exc}")

    try:
        coords = parse_coordinate_form(request.form)
        if bounds_metadata:
            b = bounds_metadata['bounds']
            if not (b['left'] <= coords['ulx'] <= b['right'] and b['left'] <= coords['lrx'] <= b['right'] and
                    b['bottom'] <= coords['lry'] <= b['top'] and b['bottom'] <= coords['uly'] <= b['top']):
                raise ValueError(
                    f"Coordinates must stay within longitude {b['left']:.6f} … {b['right']:.6f} and latitude {b['bottom']:.6f} … {b['top']:.6f}."
                )
        crop_summary = crop_geotiff(
            input_file,
            output_file,
            coords['ulx'],
            coords['uly'],
            coords['lrx'],
            coords['lry'],
        )
    except ValueError as exc:
        return render_template('index.html', error_message=str(exc), form_values=request.form, bounds_metadata=bounds_metadata)

    return render_template('view_terrain.html', file_path='cropped_' + request.form['file_path'], metadata=crop_summary)

@app.route('/process', methods=['POST'])
def process():
    if 'tiff-file' not in request.files:
        return 'No file part', 400

    file = request.files['tiff-file']
    if file.filename == '':
        return 'No selected file', 400

    ulx_geo = request.form.get('ulx_geo')
    uly_geo = request.form.get('uly_geo')
    lrx_geo = request.form.get('lrx_geo')
    lry_geo = request.form.get('lry_geo')

    if file and ulx_geo and uly_geo and lrx_geo and lry_geo:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        bounds_metadata = None
        try:
            bounds_metadata = get_raster_metadata(file_path, include_preview=True)
        except Exception as exc:
            print(f"Warning: unable to extract metadata in process: {exc}")

        output_file = 'cropped_' + file.filename
        try:
            coords = parse_coordinate_form(request.form)
            if bounds_metadata:
                b = bounds_metadata['bounds']
                if not (b['left'] <= coords['ulx'] <= b['right'] and b['left'] <= coords['lrx'] <= b['right'] and
                        b['bottom'] <= coords['lry'] <= b['top'] and b['bottom'] <= coords['uly'] <= b['top']):
                    raise ValueError(
                        f"Coordinates must stay within longitude {b['left']:.6f} … {b['right']:.6f} and latitude {b['bottom']:.6f} … {b['top']:.6f}."
                    )
            crop_summary = crop_geotiff(
                file_path,
                os.path.join(app.config['CROPPED_FOLDER'], output_file),
                coords['ulx'],
                coords['uly'],
                coords['lrx'],
                coords['lry'],
            )
        except ValueError as exc:
            return render_template('index.html', error_message=str(exc), form_values=request.form, bounds_metadata=bounds_metadata)

        return render_template('view_terrain.html', file_path=output_file, metadata=crop_summary)
    
    return 'Failed to process the file', 400

@app.route('/heightmap', methods=['GET'])
def serve_heightmap():
    file_path = request.args.get('file_path')
    if not file_path:
        return 'Heightmap file path not provided', 400
    
    absolute_file_path = os.path.join(app.config['CROPPED_FOLDER'], file_path)
    
    if not os.path.isfile(absolute_file_path):
        return 'Heightmap file not found', 404

    try:
        with rasterio.open(absolute_file_path) as dataset:
            original_height, original_width = dataset.height, dataset.width

            bounds = dataset.bounds
            transform = dataset.transform
            nodata = dataset.nodata

            max_dimension = 2048
            downsampled = False

            if original_width > max_dimension or original_height > max_dimension:
                scale_factor = max(original_width / max_dimension, original_height / max_dimension)
                new_width = int(original_width / scale_factor)
                new_height = int(original_height / scale_factor)
                heightmap_array = dataset.read(
                    1,
                    out_shape=(new_height, new_width),
                    resampling=rasterio.enums.Resampling.bilinear,
                    masked=True,
                )
                downsampled = True
                print(
                    f"Downsampled heightmap from {original_width}x{original_height} to {new_width}x{new_height} (factor: {scale_factor:.2f}x)"
                )
            else:
                heightmap_array = dataset.read(1, masked=True)
                new_width, new_height = original_width, original_height
                print(f"Serving heightmap at original resolution: {new_width}x{new_height}")

            stats = build_elevation_stats(heightmap_array)
            fill_value = stats['min'] if not np.isnan(stats['min']) else 0.0
            filled_array = heightmap_array.filled(fill_value)

            heightmap_list = filled_array.tolist()
            height, width = filled_array.shape

            return jsonify({
                'heightmap': heightmap_list,
                'width': width,
                'height': height,
                'original_width': original_width,
                'original_height': original_height,
                'bounds': {
                    'left': bounds.left,
                    'bottom': bounds.bottom,
                    'right': bounds.right,
                    'top': bounds.top
                },
                'transform': list(transform)[:6] if transform else None,
                'crs': str(dataset.crs) if dataset.crs else None,
                'downsampled': downsampled,
                'elevation_stats': stats,
                'nodata': nodata,
                'fill_value': fill_value,
            })
    except Exception as e:
        print(f"Error serving heightmap: {str(e)}")
        import traceback
        traceback.print_exc()
        return f'Failed to open heightmap file: {str(e)}', 500

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    snapshot = request.files['snapshot']
    camera_position = request.form['cameraPosition']
    camera_rotation = request.form['cameraRotation']
    sun_position = request.form['sunPosition']

    snapshot_path = os.path.join(app.config['SNAPSHOTS_FOLDER'], 'snapshot.png')
    snapshot.save(snapshot_path)

    with Image.open(snapshot_path) as img:
        img = img.convert('RGB')
        img_array = np.array(img)

    geospatial_info = {
        "transform": [123.45, 1, 0, 678.90, 0, -1],  # Example values, replace with your own
        "crs": 'EPSG:4326'
    }

    tiff_path = os.path.join(app.config['SNAPSHOTS_FOLDER'], 'snapshot.tif')
    with rasterio.open(
        tiff_path,
        'w',
        driver='GTiff',
        height=img_array.shape[0],
        width=img_array.shape[1],
        count=3,
        dtype=img_array.dtype,
        crs=geospatial_info['crs'],
        transform=from_origin(*geospatial_info['transform'])
    ) as dst:
        dst.write(img_array[:, :, 0], 1)
        dst.write(img_array[:, :, 1], 2)
        dst.write(img_array[:, :, 2], 3)

    return jsonify({'message': 'Snapshot saved successfully'})

def crop_geotiff(input_file: str, output_file: str, ulx_geo: float, uly_geo: float, lrx_geo: float, lry_geo: float) -> Dict[str, Any]:
    """Crop a GeoTIFF using rasterio with maximum precision and return metadata summary."""
    try:
        with rasterio.open(input_file) as src:
            print(f"Original file bounds: {src.bounds}")
            print(f"Original dimensions: {src.width}x{src.height}")
            print(f"Crop coordinates: UL({ulx_geo}, {uly_geo}) LR({lrx_geo}, {lry_geo})")

            left, right = ulx_geo, lrx_geo
            top, bottom = uly_geo, lry_geo

            if left > right:
                left, right = right, left
            if bottom > top:
                bottom, top = top, bottom

            bounds = src.bounds
            if right < bounds.left or left > bounds.right or top < bounds.bottom or bottom > bounds.top:
                raise ValueError(
                    "Requested crop is completely outside the raster extent."
                )

            window = from_bounds(left, bottom, right, top, src.transform)
            window = window.round_lengths().round_offsets()
            window = window.intersection(Window(0, 0, src.width, src.height))

            if window.width <= 0 or window.height <= 0:
                raise ValueError("Invalid cropping coordinates result in an empty window. Please adjust the bounds.")

            print(
                "Crop window (pixels): col_off=%s, row_off=%s, width=%s, height=%s"
                % (window.col_off, window.row_off, window.width, window.height)
            )

            data = src.read(window=window, masked=True)
            crop_transform = src.window_transform(window)
            crop_bounds = BoundingBox(*src.window_bounds(window))
            print(f"Actual crop bounds: {crop_bounds}")
            print(f"Cropped dimensions: {int(window.width)}x{int(window.height)}")

            out_meta = src.meta.copy()
            out_meta.update({
                'height': int(window.height),
                'width': int(window.width),
                'transform': crop_transform
            })

            with rasterio.open(output_file, 'w', **out_meta) as dst:
                dst.write(data.filled(src.nodata if src.nodata is not None else 0))

                if src.nodata is not None:
                    dst.nodata = src.nodata

                for i in range(1, src.count + 1):
                    description = src.descriptions[i-1] if src.descriptions and len(src.descriptions) >= i else ''
                    dst.set_band_description(i, description)

            elevation_band = data[0]
            stats = build_elevation_stats(elevation_band)

            print(f"✓ Cropped area saved as {output_file}")
            return {
                'file_path': output_file,
                'width': int(window.width),
                'height': int(window.height),
                'bounds': {
                    'left': crop_bounds.left,
                    'bottom': crop_bounds.bottom,
                    'right': crop_bounds.right,
                    'top': crop_bounds.top
                },
                'transform': list(crop_transform)[:6],
                'crs': str(src.crs) if src.crs else None,
                'resolution': [abs(crop_transform.a), abs(crop_transform.e)],
                'elevation_stats': stats,
                'nodata': src.nodata,
            }

    except Exception as e:
        print(f"✗ Error during cropping: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    app.run(debug=True)
