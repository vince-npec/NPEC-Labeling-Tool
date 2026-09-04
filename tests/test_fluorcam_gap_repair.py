from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.fluorcam_gap_repair import (
    assign_repaired_pixels_to_root_classes,
    choose_fluorcam_mirror_orientation,
    FluorCamTarMetadata,
    FluorCamAlignment,
    read_fluorcam_tar_metadata,
    repair_root_gaps_with_fluorescence,
    resolve_fluorcam_alignment,
)


class FluorCamGapRepairTests(unittest.TestCase):
    def test_read_fluorcam_tar_metadata_parses_filter_and_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "sample_FC1_FcTar.tar"
            with tarfile.open(tar_path, "w") as tf:
                files = {
                    "./protocol": b"TS=4000ms\nFilter=F635\n",
                    "./info\\device/filter-names.json": json.dumps(["F469", "F483", "F513", "F565", "F586", "F635", "Glass"]).encode("utf-8"),
                    "./info\\camera/filter-offsets.json": json.dumps(
                        [
                            {"Index": 0, "OffsetX": -2, "OffsetY": -2},
                            {"Index": 1, "OffsetX": -2, "OffsetY": 4},
                            {"Index": 2, "OffsetX": 6, "OffsetY": 0},
                            {"Index": 5, "OffsetX": 0, "OffsetY": -1},
                        ]
                    ).encode("utf-8"),
                    "./info\\camera/fish-eye.json": json.dumps({"B": -0.12, "K1": 0.0012, "K2": 2.9}).encode("utf-8"),
                }
                for name, payload in files.items():
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))

            metadata = read_fluorcam_tar_metadata(tar_path)
            self.assertEqual(metadata.filter_name, "F635")
            self.assertEqual(metadata.filter_offsets["F483"], (-2, 4))
            self.assertEqual(metadata.filter_offsets["F513"], (6, 0))
            self.assertEqual(metadata.filter_offsets["F635"], (0, -1))
            assert metadata.fish_eye is not None
            self.assertAlmostEqual(metadata.fish_eye["K2"], 2.9)

    def test_resolve_alignment_uses_known_script_preset_when_available(self) -> None:
        metadata = FluorCamTarMetadata(
            filter_name="F483",
            filter_offsets={"F483": (-2, 4), "F513": (6, 0)},
        )
        alignment = resolve_fluorcam_alignment(root_type="ROOT1", fluor_channel="FC1", metadata=metadata)
        self.assertEqual(alignment.resized_shape, (3005, 4201))
        self.assertEqual((alignment.offset_x, alignment.offset_y), (-1, 9))
        self.assertEqual(alignment.source, "hades-script-preset")

    def test_resolve_alignment_derives_unknown_filter_from_metadata_delta(self) -> None:
        metadata = FluorCamTarMetadata(
            filter_name="F635",
            filter_offsets={"F513": (6, 0), "F635": (0, -1)},
        )
        alignment = resolve_fluorcam_alignment(root_type="ROOT1", fluor_channel="FC1", metadata=metadata)
        self.assertEqual(alignment.resized_shape, (3005, 4201))
        self.assertEqual((alignment.offset_x, alignment.offset_y), (-6, -1))
        self.assertEqual(alignment.source, "preset-plus-filter-offset-delta")

    def test_choose_mirror_orientation_prefers_best_overlap_with_root_image(self) -> None:
        root_gray = np.full((80, 120), 20, dtype=np.uint8)
        yy, xx = np.ogrid[:80, :120]
        plate = ((yy - 40) ** 2 + (xx - 60) ** 2) <= 34**2
        root_gray[plate] = 230
        root_gray[15:70, 30:36] = 30
        root_gray[12:22, 30:44] = 40

        frame = np.zeros((80, 120), dtype=np.float32)
        frame[15:70, 30:36] = 1.0
        frame[12:22, 30:44] = 0.7

        alignment = FluorCamAlignment(
            target_size=(80, 120),
            resized_shape=(80, 120),
            offset_x=0,
            offset_y=0,
            root_type="ROOT1",
            fluor_channel="FC1",
            filter_name="F635",
            source="test",
        )

        mirror, source = choose_fluorcam_mirror_orientation(
            frame,
            alignment=alignment,
            root_gray=root_gray,
        )
        self.assertFalse(mirror)
        self.assertEqual(source, "auto-no-flip")

    def test_repair_root_gaps_with_fluorescence_bridges_simple_gap(self) -> None:
        root_mask = np.zeros((40, 60), dtype=np.uint8)
        root_mask[20, 10:26] = 1
        root_mask[20, 32:46] = 1

        support = np.zeros((40, 60), dtype=np.float32)
        support[20, 10:46] = 0.95
        support[19:22, 24:34] = 0.85

        repaired, bridges = repair_root_gaps_with_fluorescence(
            root_mask,
            support,
            max_gap_px=16,
            min_mean_support=0.5,
            min_endpoint_facing=0.3,
            max_bridges=2,
        )

        self.assertGreaterEqual(len(bridges), 1)
        self.assertTrue(bool(np.all(repaired[20, 10:46] > 0)))

    def test_assign_repaired_pixels_prefers_nearest_existing_root_class(self) -> None:
        pred = np.zeros((20, 24), dtype=np.uint8)
        pred[10, 3:8] = 1
        pred[10, 16:21] = 3
        repaired = np.zeros_like(pred, dtype=np.uint8)
        repaired[10, 3:21] = 1

        updated, root_added, lateral_added = assign_repaired_pixels_to_root_classes(
            pred,
            repaired,
            root_class_id=1,
            lateral_class_id=3,
        )

        self.assertEqual(root_added + lateral_added, 8)
        self.assertTrue(bool(np.all(updated[10, 8:12] == 1)))
        self.assertTrue(bool(np.all(updated[10, 12:16] == 3)))


if __name__ == "__main__":
    unittest.main()
