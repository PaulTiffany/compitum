import pytest

from compitum.trajectory import blockwise_audit

BLOCKS = {"hidden": (0, 2), "latent": (2, 4)}


def test_transport_lesson_product_breach_with_contractive_hidden_block() -> None:
    """The load-bearing Sketched lesson: product-metric growth caused purely
    by transport into an initially unperturbed block, while the perturbed
    block itself contracts."""
    base = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    # hidden perturbation contracts 1.0 -> 0.9; latent emerges 0.0 -> 1.0
    probe = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 1.0, 0.0]]
    audit = blockwise_audit(base, probe, BLOCKS)
    first = audit["first_step"]
    assert first["product_breach"] is True
    assert first["block_breaches"]["hidden"] is False
    assert first["block_emergence"]["latent"] is True
    step = audit["steps"][0]
    assert step["blocks"]["hidden"]["gain"] == pytest.approx(0.9)
    assert step["blocks"]["latent"]["gain"] is None  # emerged from zero
    assert step["product_l2_gain"] == pytest.approx((0.9**2 + 1.0**2) ** 0.5)
    assert "transport across a block interface" in audit["method"]["interpretation_boundary"]


def test_uniform_contraction_has_no_breach_anywhere() -> None:
    base = [[0.0] * 4, [0.0] * 4]
    probe = [[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]]
    audit = blockwise_audit(base, probe, BLOCKS)
    first = audit["first_step"]
    assert first["product_breach"] is False
    assert first["max_block_breach"] is False
    assert first["block_breaches"] == {"hidden": False, "latent": False}


def test_validation_rejections() -> None:
    good = [[0.0] * 4, [0.0] * 4]
    with pytest.raises(ValueError, match="equal length >= 2"):
        blockwise_audit([good[0]], [good[0]], BLOCKS)
    with pytest.raises(ValueError, match="share one dimension"):
        blockwise_audit(good, [[0.0] * 4, [0.0] * 3], BLOCKS)
    with pytest.raises(ValueError, match="nonempty"):
        blockwise_audit([[], []], [[], []], {})
    with pytest.raises(ValueError, match="at least one block"):
        blockwise_audit(good, good, {})
    with pytest.raises(ValueError, match="out of bounds"):
        blockwise_audit(good, good, {"a": (0, 5)})
    with pytest.raises(ValueError, match="out of bounds"):
        blockwise_audit(good, good, {"a": (2, 2), "b": (0, 4)})
    with pytest.raises(ValueError, match="overlaps"):
        blockwise_audit(good, good, {"a": (0, 3), "b": (2, 4)})
    with pytest.raises(ValueError, match="cover the full state dimension"):
        blockwise_audit(good, good, {"a": (0, 2)})
    with pytest.raises(ValueError, match="non-finite"):
        blockwise_audit([[float("nan")] * 4, [0.0] * 4], [[0.0] * 4, [0.0] * 4], BLOCKS)
