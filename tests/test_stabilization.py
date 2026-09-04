from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from resources.stabilization import (
    StabilizationConfig,
    StabilizationResult,
    build_registration_mask,
    next_stabilization_reference,
    stabilize_against_reference,
)


class StabilizationTests(unittest.TestCase):
    def _synthetic_plate(
        self,
        plant_shift: tuple[int, int] = (0, 0),
        *,
        root_shift: tuple[int, int] | None = None,
        shoot_axes: tuple[int, int] = (18, 9),
    ) -> np.ndarray:
        height, width = 320, 420
        image = np.full((height, width, 3), 225, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (width - 20, height - 20), (90, 92, 95), 3)
        shift_x, shift_y = plant_shift
        root_shift_x, root_shift_y = root_shift if root_shift is not None else plant_shift
        for center_x in (150, 210, 270):
            cv2.ellipse(
                image,
                (center_x + shift_x, 105 + shift_y),
                shoot_axes,
                0,
                0,
                360,
                (72, 122, 38),
                -1,
            )
            cv2.line(
                image,
                (center_x + root_shift_x, 115 + root_shift_y),
                (center_x + root_shift_x + 4, 240 + root_shift_y),
                (105, 110, 92),
                2,
            )
        return image

    def _textured_lucifer_plate(self) -> np.ndarray:
        height, width = 720, 820
        image = np.full((height, width, 3), 226, dtype=np.uint8)
        cv2.rectangle(image, (24, 20), (width - 25, height - 24), (82, 86, 91), 5)
        cv2.rectangle(image, (38, 34), (width - 39, height - 38), (176, 180, 185), 2)
        cv2.putText(image, "LUCIFER 5", (56, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 72, 75), 2)
        for index, center_x in enumerate((130, 270, 410, 550, 690)):
            center_y = 150 + ((index % 2) * 9)
            cv2.ellipse(image, (center_x - 13, center_y), (24, 12), -18, 0, 360, (59, 122, 38), -1)
            cv2.ellipse(image, (center_x + 17, center_y - 7), (21, 10), 22, 0, 360, (68, 137, 43), -1)
            cv2.circle(image, (center_x, center_y + 7), 5, (48, 92, 31), -1)
            cv2.line(image, (center_x, center_y + 9), (center_x + 5, 455), (104, 111, 92), 3)
            for branch_index, branch_y in enumerate((235, 300, 365, 420)):
                direction = -1 if (index + branch_index) % 2 else 1
                cv2.line(
                    image,
                    (center_x + 3, branch_y),
                    (center_x + (direction * (35 + (branch_index * 5))), branch_y + 20),
                    (112, 116, 99),
                    2,
                )
        random = np.random.default_rng(20260722)
        for x, y, radius, shade in zip(
            random.integers(70, width - 70, size=110),
            random.integers(70, 450, size=110),
            random.integers(1, 4, size=110),
            random.integers(120, 205, size=110),
        ):
            value = int(shade)
            cv2.circle(image, (int(x), int(y)), int(radius), (value, value + 2, value - 3), -1)
        return image

    def _warp_plate(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return cv2.warpAffine(
            image,
            np.asarray(matrix, dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _assert_identity_noop(
        self,
        result: StabilizationResult,
        moving: np.ndarray,
        expected_status: str,
    ) -> None:
        np.testing.assert_array_equal(result.warped_image, moving)
        np.testing.assert_array_equal(
            result.warp_matrix,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(float(result.shift_x), 0.0)
        self.assertEqual(float(result.shift_y), 0.0)
        self.assertEqual(float(result.rotation_deg), 0.0)
        self.assertTrue(np.isfinite(result.score))
        self.assertEqual(result.method, "identity")
        self.assertEqual(result.status, expected_status)
        self.assertFalse(result.accepted)

    def test_static_dish_with_shifted_shoots_is_identity_noop(self) -> None:
        reference = self._synthetic_plate()
        moving = self._synthetic_plate(plant_shift=(18, -7), root_shift=(0, 0))

        plant_result = stabilize_against_reference(
            reference,
            moving,
            StabilizationConfig(target_mode="plant_top", reference_mode="first", downscale_max_dim=420),
        )
        frame_result = stabilize_against_reference(
            reference,
            moving,
            StabilizationConfig(target_mode="dish_frame", reference_mode="first", downscale_max_dim=420),
        )

        self._assert_identity_noop(
            plant_result,
            moving,
            "rejected_biological_motion_conflict",
        )
        self.assertLess(abs(float(frame_result.shift_x)), 1.0)
        self.assertLess(abs(float(frame_result.shift_y)), 1.0)

    def test_static_dish_with_asymmetric_shoot_growth_is_identity_noop(self) -> None:
        reference = self._synthetic_plate(shoot_axes=(10, 5), root_shift=(0, 0))
        moving = self._synthetic_plate(
            plant_shift=(10, -3),
            root_shift=(0, 0),
            shoot_axes=(34, 15),
        )
        config = StabilizationConfig(
            target_mode="plant_top",
            reference_mode="first",
            downscale_max_dim=420,
            enable_feature_fallback=False,
        )

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            return_value=((12.0, -4.0), 0.78),
        ):
            result = stabilize_against_reference(reference, moving, config)

        self._assert_identity_noop(
            result,
            moving,
            "rejected_biological_motion_conflict",
        )

    def test_true_camera_translation_is_not_vetoed(self) -> None:
        reference = self._synthetic_plate()
        forward = np.array([[1.0, 0.0, 18.0], [0.0, 1.0, -7.0]], dtype=np.float32)
        moving = self._warp_plate(reference, forward)

        result = stabilize_against_reference(
            reference,
            moving,
            StabilizationConfig(target_mode="plant_top", reference_mode="first", downscale_max_dim=420),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "plant_top_phase_correlation")
        self.assertAlmostEqual(float(result.shift_x), -18.0, delta=1.5)
        self.assertAlmostEqual(float(result.shift_y), 7.0, delta=1.5)

    def test_small_rotation_and_translation_use_affine_candidate(self) -> None:
        reference = self._textured_lucifer_plate()
        height, width = reference.shape[:2]
        forward = cv2.getRotationMatrix2D((0.5 * width, 0.5 * height), 2.2, 1.0)
        forward[:, 2] += np.array([15.0, -9.0], dtype=np.float64)
        moving = self._warp_plate(reference, forward)

        result = stabilize_against_reference(
            reference,
            moving,
            StabilizationConfig(target_mode="plant_top", downscale_max_dim=410),
        )

        expected = cv2.invertAffineTransform(forward)
        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "plant_top_orb_ransac_affine")
        self.assertAlmostEqual(float(result.rotation_deg), 2.2, delta=0.35)
        np.testing.assert_allclose(result.warp_matrix[:, :2], expected[:, :2], atol=0.025)
        np.testing.assert_allclose(result.warp_matrix[:, 2], expected[:, 2], atol=3.0)
        mask = build_registration_mask(reference.shape[:2], StabilizationConfig(target_mode="plant_top")) > 0
        before_error = float(np.mean(np.abs(reference.astype(np.float32) - moving.astype(np.float32))[mask]))
        after_error = float(
            np.mean(np.abs(reference.astype(np.float32) - result.warped_image.astype(np.float32))[mask])
        )
        self.assertLess(after_error, 0.45 * before_error)

    def test_feature_affine_falls_back_when_phase_response_is_weak(self) -> None:
        reference = self._textured_lucifer_plate()
        forward = np.array([[1.0, 0.0, 17.0], [0.0, 1.0, -11.0]], dtype=np.float32)
        moving = self._warp_plate(reference, forward)
        config = StabilizationConfig(target_mode="plant_top", downscale_max_dim=820)

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            return_value=((0.0, 0.0), 0.01),
        ):
            result = stabilize_against_reference(reference, moving, config)

        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "plant_top_orb_ransac_affine")
        self.assertAlmostEqual(float(result.shift_x), -17.0, delta=2.0)
        self.assertAlmostEqual(float(result.shift_y), 11.0, delta=2.0)
        self.assertAlmostEqual(float(result.rotation_deg), 0.0, delta=0.25)
        self.assertGreater(float(result.score), float(config.min_affine_alignment_score))

    def test_excessive_affine_rotation_is_rejected_as_identity(self) -> None:
        reference = self._textured_lucifer_plate()
        height, width = reference.shape[:2]
        forward = cv2.getRotationMatrix2D((0.5 * width, 0.5 * height), 10.0, 1.0)
        moving = self._warp_plate(reference, forward)
        config = StabilizationConfig(
            target_mode="plant_top",
            downscale_max_dim=820,
            max_rotation_deg=4.0,
        )

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            return_value=((0.0, 0.0), 0.01),
        ):
            result = stabilize_against_reference(reference, moving, config)

        self._assert_identity_noop(result, moving, "rejected_large_rotation")

    def test_stable_frame_phase_fallback_recovers_rejected_plant_translation(self) -> None:
        reference = self._textured_lucifer_plate()
        forward = np.array([[1.0, 0.0, 17.0], [0.0, 1.0, -11.0]], dtype=np.float32)
        moving = self._warp_plate(reference, forward)
        config = StabilizationConfig(
            target_mode="plant_top",
            downscale_max_dim=820,
            enable_feature_fallback=False,
            enable_stable_frame_affine_fallback=False,
        )

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            side_effect=[((0.0, 0.0), 0.01), ((17.0, -11.0), 0.45)],
        ):
            result = stabilize_against_reference(reference, moving, config)

        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "dish_frame_phase_fallback")
        self.assertAlmostEqual(float(result.shift_x), -17.0, delta=0.25)
        self.assertAlmostEqual(float(result.shift_y), 11.0, delta=0.25)
        self.assertEqual(float(result.rotation_deg), 0.0)
        self.assertAlmostEqual(float(result.score), 0.45, delta=1.0e-6)

    def test_stable_frame_affine_fallback_handles_small_camera_rotation(self) -> None:
        reference = self._textured_lucifer_plate()
        height, width = reference.shape[:2]
        forward = cv2.getRotationMatrix2D((0.5 * width, 0.5 * height), 1.4, 1.0)
        forward[:, 2] += np.array([11.0, -6.0], dtype=np.float64)
        moving = self._warp_plate(reference, forward)
        config = StabilizationConfig(
            target_mode="plant_top",
            downscale_max_dim=410,
            enable_feature_fallback=False,
        )

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            side_effect=[((0.0, 0.0), 0.01), ((0.0, 0.0), 0.01)],
        ):
            result = stabilize_against_reference(reference, moving, config)

        expected = cv2.invertAffineTransform(forward)
        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "dish_frame_orb_ransac_affine_fallback")
        self.assertAlmostEqual(float(result.rotation_deg), 1.4, delta=0.35)
        np.testing.assert_allclose(result.warp_matrix[:, :2], expected[:, :2], atol=0.025)
        np.testing.assert_allclose(result.warp_matrix[:, 2], expected[:, 2], atol=3.0)

    def test_stable_frame_fallback_rejects_growth_driven_bogus_shift(self) -> None:
        reference = self._synthetic_plate()
        moving = self._synthetic_plate(plant_shift=(28, 0))
        config = StabilizationConfig(
            target_mode="plant_top",
            downscale_max_dim=420,
            enable_feature_fallback=False,
            enable_stable_frame_affine_fallback=False,
        )

        with patch(
            "resources.stabilization.cv2.phaseCorrelate",
            side_effect=[((0.0, 0.0), 0.01), ((35.0, 0.0), 0.90)],
        ):
            result = stabilize_against_reference(reference, moving, config)

        self._assert_identity_noop(result, moving, "rejected_low_response")

    def test_low_texture_frames_are_rejected_as_identity_noops(self) -> None:
        reference = np.full((320, 420, 3), 225, dtype=np.uint8)
        moving = np.full_like(reference, 220)

        for target_mode in ("plant_top", "dish_frame"):
            with self.subTest(target_mode=target_mode):
                result = stabilize_against_reference(
                    reference,
                    moving,
                    StabilizationConfig(target_mode=target_mode, downscale_max_dim=420),
                )

                self._assert_identity_noop(result, moving, "rejected_low_response")

    def test_non_finite_phase_correlation_output_is_rejected(self) -> None:
        reference = self._synthetic_plate()
        moving = self._synthetic_plate(plant_shift=(8, -3))
        config = StabilizationConfig(downscale_max_dim=420)

        invalid_outputs = (
            ((float("nan"), 0.0), 0.9),
            ((0.0, float("inf")), 0.9),
            ((0.0, 0.0), float("nan")),
        )
        for phase_output in invalid_outputs:
            with self.subTest(phase_output=phase_output):
                with patch("resources.stabilization.cv2.phaseCorrelate", return_value=phase_output):
                    result = stabilize_against_reference(reference, moving, config)

                self._assert_identity_noop(result, moving, "rejected_non_finite")

    def test_large_high_response_shift_is_rejected_as_an_outlier(self) -> None:
        reference = self._synthetic_plate()
        moving = self._synthetic_plate(plant_shift=(100, 0))
        config = StabilizationConfig(downscale_max_dim=420)

        result = stabilize_against_reference(reference, moving, config)

        self.assertGreater(float(result.score), float(config.min_response))
        self._assert_identity_noop(result, moving, "rejected_large_shift")

    def test_rejected_outlier_does_not_poison_previous_frame_chain(self) -> None:
        config = StabilizationConfig(reference_mode="previous", downscale_max_dim=420)
        reference = self._synthetic_plate()
        outlier = self._synthetic_plate(plant_shift=(100, 0))

        rejected = stabilize_against_reference(reference, outlier, config)
        next_reference = next_stabilization_reference(reference, rejected, config)

        self.assertFalse(rejected.accepted)
        self.assertIs(next_reference, reference)

        valid_next_frame = self._warp_plate(
            reference,
            np.array([[1.0, 0.0, 20.0], [0.0, 1.0, -6.0]], dtype=np.float32),
        )
        recovered = stabilize_against_reference(next_reference, valid_next_frame, config)

        self.assertTrue(recovered.accepted)
        self.assertAlmostEqual(float(recovered.shift_x), -20.0, delta=1.5)
        self.assertAlmostEqual(float(recovered.shift_y), 6.0, delta=1.5)
        self.assertIs(next_stabilization_reference(next_reference, recovered, config), recovered.warped_image)

    def test_registration_mask_modes_use_different_areas(self) -> None:
        plant_mask = build_registration_mask((320, 420), StabilizationConfig(target_mode="plant_top"))
        frame_mask = build_registration_mask((320, 420), StabilizationConfig(target_mode="dish_frame"))

        self.assertGreater(int(np.count_nonzero(plant_mask[40:180, 80:340])), 0)
        self.assertEqual(int(plant_mask[300, 210]), 0)
        self.assertGreater(int(frame_mask[300, 210]), 0)
        self.assertGreater(int(frame_mask[160, 10]), 0)


if __name__ == "__main__":
    unittest.main()
