# Camera URL configuration via environment

The admin page (`/admin` on Camera 6) reads camera base URLs and paths from environment variables.

Quick start (PowerShell example):

```
$env:CAM_IDS = "102,202,302,402,502,602,702,Transshipment"

# Example domains
$env:CAM_102_DOMAIN = "https://jswvid.fidelisam.in"
$env:CAM_102_PATH = "/cam1"
$env:CAM_102_RTSP_URL = "rtsp://user:pass@ip:port/Streaming/Channels/102"

$env:CAM_502_DOMAIN = "https://jswvido.fidelisam.in"
$env:CAM_502_PATH = "/cam5"
$env:CAM_502_RTSP_URL = "rtsp://user:pass@ip:port/Streaming/Channels/502"

# Optional overrides
$env:CAM_502_LOGICAL_ID = "cam5"
$env:CAM_502_NAME = "Loading Bay Camera 5"
$env:CAM_502_LOCATION = "Entry Point 5"
```

Alternatively, supply a JSON blob in `CAMERA_CONFIGS_JSON` that maps camera IDs to `{ domain, url_path, logical_id, name, location }`.

```
$env:CAMERA_CONFIGS_JSON = '{
  "502": {"domain":"https://jswvido.fidelisam.in","url_path":"/cam5","logical_id":"cam5","name":"Cam 5","location":"Bay 5"}
}'
```

If nothing is specified, defaults for IDs 102–702 and Transshipment are used.


