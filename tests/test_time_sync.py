"""Tests for solution.utils.time_sync alignment module."""

import numpy as np
import pytest

from solution.utils.time_sync import (
    AlignmentResult,
    imu_timestamp_to_seconds,
    frame_index_to_seconds,
    seconds_to_frame_index,
    imu_sample_to_frame,
    frame_to_imu_sample,
    compute_alignment_cross_correlation,
    compute_alignment_default,
    align_imu_to_video,
    get_imu_window,
    _normalize,
)


class TestTimestampConversion:
    def test_imu_timestamp_to_seconds_basic(self):
        ts = np.array([1e18, 1e18 + 1e9, 1e18 + 2e9])
        result = imu_timestamp_to_seconds(ts)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_imu_timestamp_to_seconds_empty(self):
        result = imu_timestamp_to_seconds(np.array([]))
        assert len(result) == 0

    def test_imu_timestamp_to_seconds_single(self):
        result = imu_timestamp_to_seconds(np.array([5e18]))
        np.testing.assert_allclose(result, [0.0])

    def test_frame_index_to_seconds(self):
        assert frame_index_to_seconds(0, 15.0) == 0.0
        assert frame_index_to_seconds(15, 15.0) == 1.0
        assert frame_index_to_seconds(150, 15.0) == 10.0

    def test_seconds_to_frame_index(self):
        assert seconds_to_frame_index(0.0, 15.0) == 0
        assert seconds_to_frame_index(1.0, 15.0) == 15
        assert seconds_to_frame_index(10.0, 15.0) == 150

    def test_seconds_to_frame_index_clamps_negative(self):
        assert seconds_to_frame_index(-5.0, 15.0) == 0


class TestBidirectionalMapping:
    def setup_method(self):
        self.alignment = AlignmentResult(
            offset_samples=-3,
            offset_sec=-0.2,
            correlation_score=0.5,
            method="test",
            segment_offsets=[],
        )

    def test_imu_sample_to_frame(self):
        assert imu_sample_to_frame(100, self.alignment) == 103

    def test_frame_to_imu_sample(self):
        assert frame_to_imu_sample(100, self.alignment) == 97

    def test_roundtrip(self):
        for frame_idx in [0, 50, 500, 9000]:
            sample = frame_to_imu_sample(frame_idx, self.alignment)
            back = imu_sample_to_frame(sample, self.alignment)
            assert back == frame_idx

    def test_zero_offset_identity(self):
        zero_align = AlignmentResult(
            offset_samples=0, offset_sec=0.0,
            correlation_score=1.0, method="test", segment_offsets=[],
        )
        assert imu_sample_to_frame(42, zero_align) == 42
        assert frame_to_imu_sample(42, zero_align) == 42


class TestCrossCorrelation:
    def test_identical_signals_zero_offset(self):
        np.random.seed(42)
        signal = np.random.randn(500)
        result = compute_alignment_cross_correlation(signal, signal, fps=15.0)
        assert result.offset_samples == 0
        assert result.method == "cross_correlation"

    def test_shifted_signal_detects_offset(self):
        np.random.seed(42)
        base = np.random.randn(500)
        shift = 5
        shifted = np.roll(base, shift)
        result = compute_alignment_cross_correlation(base, shifted, fps=15.0)
        assert abs(result.offset_samples - shift) <= 1

    def test_negative_shift_detected(self):
        np.random.seed(42)
        base = np.random.randn(500)
        shift = -7
        shifted = np.roll(base, shift)
        result = compute_alignment_cross_correlation(base, shifted, fps=15.0)
        assert abs(result.offset_samples - shift) <= 1

    def test_segment_offsets_returned(self):
        np.random.seed(42)
        signal = np.random.randn(1000)
        result = compute_alignment_cross_correlation(signal, signal, fps=15.0)
        assert len(result.segment_offsets) == 4
        for seg_idx, offset in result.segment_offsets:
            assert abs(offset) <= 2

    def test_mismatched_lengths_handled(self):
        np.random.seed(42)
        a = np.random.randn(500)
        b = np.random.randn(490)
        result = compute_alignment_cross_correlation(a, b, fps=15.0)
        assert result.method == "cross_correlation"


class TestDefaultAlignment:
    def test_default_values(self):
        result = compute_alignment_default(fps=15.0)
        assert result.offset_samples == -3
        assert result.offset_sec == pytest.approx(-0.2)
        assert result.method == "default"
        assert result.segment_offsets == []


class TestLegacyAlignImuToVideo:
    def test_basic_mapping(self):
        ts = np.arange(100) * 1e9 / 15.0 + 1e18
        result = align_imu_to_video(ts, 15.0, 100)
        assert len(result) == 100
        assert result[0] == 0
        assert result[3] == 0
        assert result[10] == 7

    def test_empty_inputs(self):
        result = align_imu_to_video(np.array([]), 15.0, 0)
        assert len(result) == 0

    def test_clipping(self):
        ts = np.arange(50) * 1e9 / 15.0 + 1e18
        result = align_imu_to_video(ts, 15.0, 50)
        assert result.min() >= 0
        assert result.max() < 50


class TestGetImuWindow:
    def test_center_window(self):
        imu = np.random.randn(100, 11)
        alignment = np.arange(100, dtype=np.int64)
        window = get_imu_window(imu, 50, alignment, window_size=3)
        assert window.shape == (7, 11)

    def test_edge_start(self):
        imu = np.random.randn(100, 11)
        alignment = np.arange(100, dtype=np.int64)
        window = get_imu_window(imu, 0, alignment, window_size=5)
        assert window.shape[0] == 6
        assert window.shape[1] == 11

    def test_edge_end(self):
        imu = np.random.randn(100, 11)
        alignment = np.arange(100, dtype=np.int64)
        window = get_imu_window(imu, 99, alignment, window_size=5)
        assert window.shape[0] == 6

    def test_out_of_bounds(self):
        imu = np.random.randn(100, 11)
        alignment = np.arange(100, dtype=np.int64)
        window = get_imu_window(imu, -1, alignment)
        assert window.shape[0] == 0
        window = get_imu_window(imu, 200, alignment)
        assert window.shape[0] == 0


class TestNormalize:
    def test_normalized_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _normalize(x)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_constant_signal_returns_zeros(self):
        x = np.ones(100)
        result = _normalize(x)
        np.testing.assert_allclose(result, 0.0)
