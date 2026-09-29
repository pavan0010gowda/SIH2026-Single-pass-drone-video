# PRISM · single-pass drone video → metric 3-D digital twin (SIH NTRO, problem statement 17)

PRISM turns **one drone pass** (video + the DJI .SRT flight log) into a georeferenced, metrically accurate
3-D model, and turns the model into answers: heights, roofs and facades, roads and potholes, cover and
concealment, base sites, covert routes, change detection, and standard GIS / 3-D deliverables.

## Run it

```powershell
pip install -r backend/requirements.txt
python backend/app.py            # then open http://localhost:8000
```

Heavy 4K videos: process them on a free Colab GPU with `colab/prism_turbo_reconstruction.ipynb`
(set `QUALITY = "sih"`), then **Mission ▸ New mission ▸ Colab package** and drop `prism_colab_bundle.zip`.
Short clips can be reconstructed on this computer (COLMAP 4.x with CUDA).

Accuracy tests (synthetic survey with known answers, no GPU needed): `python backend/tests/test_accuracy.py`

## How the problem statement is covered

| Requirement | Where |
|---|---|
| Terrain and structures | bare-earth DTM (progressive morphological filter), DSM, nDSM, land cover (`backend/terrain.py`) |
| Building facades and rooftops | roof planes → flat / shed / gable / hipped / complex, pitch, eaves, storeys, facade coverage, LoD 1.2 blocks (`buildings.py`, *Structures* panel) |
| Roads and infrastructure | road extraction with traffic-torn sections repaired, widths, grades, potholes with ASTM severity (`road_engine.py`) |
| Vegetation and obstacles | trees (crown-apex heights), vehicles, height-ceiling obstacles (`height_engine.py`) |
| Textured meshes / point clouds | gap-filled screened Poisson mesh (`mesh_builder.py`), dense coloured cloud |
| Dynamic objects (vehicles, people, animals) | YOLO-seg masks out of feature matching; an epipolar motion test removes only **moving** objects from depth fusion (`masks.py`, Colab engine) |
| GPS inaccuracies, no GCPs | robust Sim3 / levelled fit of the camera track to the GPS log, lat/lon order verified, uncertainty reported (`georeference.py`, `recalibrate.py`) |
| A few GCPs | *Quality ▸ Add check point*: residuals, leave-one-out RMSE, one-click bias correction |
| Optional inputs | RTK / PPK (RTKLIB .pos or CSV, clock auto-aligned), camera calibration, IMU / gimbal flight logs, barometer (`sensors.py`) |
| Processing time < 15 min for a 10-min video | Colab `QUALITY="sih"`: budget = 1.4 × video length, NVDEC in-decoder resize; the achieved ratio is in the report |
| Output formats | OBJ, PLY, **LAS**, **GeoTIFF** (DSM, DTM, nDSM, orthophoto, land cover), **glTF**, GLB, **FBX**, GeoJSON, CityJSON (*Mission ▸ Export centre*, `geo_export.py`) |
| Visualisation | web viewer with synchronised drone video and AR overlay |
| Accuracy / completeness evidence | *Analyze ▸ Quality & accuracy* and the printable report (`quality.py`) |

Coordinates: the model frame is metres, x = East, y = Up, z = South, origin on the ground at the GPS
origin. LAS / GeoTIFF / CityJSON are written in WGS 84 / UTM (exact Krüger series, no GDAL needed);
GeoJSON in WGS 84.
