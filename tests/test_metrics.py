import numpy as np

from nuclei_counting.metrics import (
    component_density,
    instance_failure_decomposition,
    remove_small_components,
)


def test_small_component_filter_and_density_integral():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:4, 1:4] = True   # area 9, keep
    mask[8:10, 8:10] = True  # area 4, remove
    labels = remove_small_components(mask, min_area=8, connectivity=2)
    assert labels.max() == 1
    density, count = component_density(labels, sigma=1.0)
    assert count == 1
    assert np.isclose(density.sum(), 1.0, atol=1e-5)


def test_merge_proxy():
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[2:5, 1:4] = 1
    gt[2:5, 5:8] = 2
    pred = np.zeros_like(gt)
    pred[2:5, 1:8] = 1
    diagnostics, _ = instance_failure_decomposition(
        pred,
        gt,
        iou_threshold=0.50,
        coverage_threshold=0.10,
    )
    assert diagnostics.merge_errors == 1
    assert diagnostics.split_errors == 0
    assert diagnostics.pred_components == 1
    assert diagnostics.gt_instances == 2


def test_split_proxy():
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[2:7, 2:8] = 1
    pred = np.zeros_like(gt)
    pred[2:7, 2:4] = 1
    pred[2:7, 6:8] = 2
    diagnostics, _ = instance_failure_decomposition(
        pred,
        gt,
        iou_threshold=0.50,
        coverage_threshold=0.10,
    )
    assert diagnostics.merge_errors == 0
    assert diagnostics.split_errors == 1
    assert diagnostics.pred_components == 2
    assert diagnostics.gt_instances == 1
