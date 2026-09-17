from pathlib import Path

import pytest

import tools.minimax_h3_cache_quality_gate as quality_gate


def test_parse_cache_records_requires_world_consensus_and_hits(tmp_path: Path) -> None:
    for rank in range(16):
        (tmp_path / f"node_{rank}.log").write_text(
            "MINIMAX_H3_CACHE_RESULT "
            f"rank={rank} enabled=1 dense_forwards=48 cache_hits=1 "
            "similarity_checks=1 block_forwards=2401 hit_forwards=5\n",
            encoding="utf-8",
        )

    result = quality_gate._parse_cache_records(tmp_path)

    assert result["consensus"]["cache_hits"] == 1
    assert result["consensus"]["block_forwards"] == 2401
    assert len(result["records"]) == 16


def test_parse_cache_records_rejects_rank_disagreement(tmp_path: Path) -> None:
    for rank in range(16):
        hits = 18 if rank == 7 else 19
        dense = 49 - hits
        hit_forwards = ",".join(str(5 + 2 * index) for index in range(hits))
        (tmp_path / f"node_{rank}.log").write_text(
            "MINIMAX_H3_CACHE_RESULT "
            f"rank={rank} enabled=1 dense_forwards={dense} cache_hits={hits} "
            f"similarity_checks=45 block_forwards={2450 - 49 * hits} "
            f"hit_forwards={hit_forwards}\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="disagree"):
        quality_gate._parse_cache_records(tmp_path)


def test_evaluate_applies_fixed_quality_thresholds() -> None:
    video = {
        "ssim": {"all_frame_mean": 0.971},
        "raw_uint8": {"all_frame_psnr_db": 34.1},
    }
    audio = {
        "centered_pearson": {"minimum": 0.981},
        "rms_ratio_native_over_official": {"min": 0.96, "max": 1.04},
        "stft_magnitude_spectral_cosine": {str(n_fft): {"minimum": 0.981} for n_fft in quality_gate.STFT_FFT_SIZES},
    }

    assert quality_gate._evaluate(video, audio)["passed"] is True
    video["ssim"]["all_frame_mean"] = 0.969
    result = quality_gate._evaluate(video, audio)
    assert result["passed"] is False
    assert result["failed_metrics"][0]["metric"] == "video.ssim"
