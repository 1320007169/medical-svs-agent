from pathlib import Path

import pytest
from PIL import Image

from medical_svs_agent.slide import OpenSlideCropService, SlideError


class FakeSlide:
    dimensions = (10000, 5000)
    level_count = 3
    level_downsamples = (1.0, 4.0, 16.0)
    properties = {"openslide.mpp-x": "0.25", "openslide.objective-power": "40"}

    def __init__(self):
        self.read = None
        self.closed = False

    def read_region(self, location, level, size):
        self.read = (location, level, size)
        return Image.new("RGBA", size, "white")

    def close(self):
        self.closed = True


def test_relative_bbox_reads_requested_pyramid_region(tmp_path: Path):
    slide_path = tmp_path / "case.svs"
    slide_path.touch()
    fake = FakeSlide()
    service = OpenSlideCropService({"case": slide_path}, opener=lambda _: fake, max_crop_side=2048)
    result, image = service.crop("case", {"bbox_2d": [100, 200, 180, 360], "level": 1})
    assert fake.read == ((1000, 1000), 1, (200, 200))
    assert image.size == (200, 200)
    assert result["level0_bbox"] == [1000, 1000, 1800, 1800]
    assert result["mpp_x"] == "0.25"
    assert fake.closed


def test_large_level_zero_crop_is_rejected(tmp_path: Path):
    slide_path = tmp_path / "case.svs"
    slide_path.touch()
    service = OpenSlideCropService(
        {"case": slide_path}, opener=lambda _: FakeSlide(), max_crop_side=512
    )
    with pytest.raises(SlideError, match="smaller bbox or a coarser level"):
        service.crop("case", {"bbox_2d": [0, 0, 1000, 1000]})


def test_unknown_slide_is_rejected():
    service = OpenSlideCropService({"known": Path("/tmp/known.svs")}, opener=lambda _: FakeSlide())
    with pytest.raises(SlideError, match="unknown slide_id"):
        service.crop("other", {"bbox_2d": [0, 0, 10, 10]})

