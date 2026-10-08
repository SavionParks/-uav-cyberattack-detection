"""Download three public-domain/CC0 traffic-sign demo images from Wikimedia Commons."""

from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent / "demo_images"
OUT.mkdir(exist_ok=True)

IMAGES = {
    "stop.png": "https://thumb.wikimedia.org/wikipedia/commons/thumb/e/ed/MUTCD_Stop_Sign_%28R1-1%29.svg/500px-MUTCD_Stop_Sign_%28R1-1%29.svg.png",
    "speed_limit_35.png": "https://thumb.wikimedia.org/wikipedia/commons/thumb/2/26/Speed_Limit_35_sign.svg/960px-Speed_Limit_35_sign.svg.png",
    "speed_limit_45.png": "https://thumb.wikimedia.org/wikipedia/commons/thumb/6/6d/Speed_Limit_45_sign.svg/960px-Speed_Limit_45_sign.svg.png",
}

for filename, url in IMAGES.items():
    destination = OUT / filename
    request = Request(url, headers={"User-Agent": "uav-cyberattack-detection-class-demo/1.0"})
    with urlopen(request, timeout=30) as response:
        destination.write_bytes(response.read())
    print("Downloaded:", destination)

print("Done. The files are in:", OUT)
