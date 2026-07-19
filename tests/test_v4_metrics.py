import torch

from utils.v4_metrics import edge_metrics, thin_structure_metrics, unsupported_new_edge_rate


def test_identical_edge_and_skeleton_metrics_are_perfect():
    image = torch.zeros(3, 16, 16)
    image[:, 4:12, 7:9] = 1.0
    edges = edge_metrics(image, image)
    thin = thin_structure_metrics(image, image)
    assert edges["edge_f1"] > 0.999
    assert thin["skeleton_f1"] > 0.999
    assert unsupported_new_edge_rate(image, image, torch.zeros(1, 16, 16)) == 0.0
