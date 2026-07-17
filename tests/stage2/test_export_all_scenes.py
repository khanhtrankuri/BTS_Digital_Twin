from argparse import Namespace
import json

from tools.export_stage2_dataset import discover_scene_dirs, rebase_scene_samples, scene_arguments


def make_phase1_scene(root, name):
    scene = root / name
    (scene / "train" / "sparse").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    (scene / "test" / "test_poses.csv").write_text("image_name\n", encoding="utf-8")
    return scene


def test_parent_discovers_all_scenes_in_sorted_order(tmp_path):
    make_phase1_scene(tmp_path, "HNI0366")
    make_phase1_scene(tmp_path, "HNI0131")
    (tmp_path / "not_a_scene").mkdir()
    scenes, parent_mode = discover_scene_dirs(tmp_path)
    assert parent_mode
    assert [scene.name for scene in scenes] == ["HNI0131", "HNI0366"]


def test_direct_scene_preserves_single_scene_mode(tmp_path):
    scene = make_phase1_scene(tmp_path, "HNI0265")
    scenes, parent_mode = discover_scene_dirs(scene)
    assert not parent_mode and scenes == [scene.resolve()]


def test_combined_manifest_paths_are_rebased_per_scene():
    sample = {"rgb_render": "train/rgb_render/0.png", "depth": "train/depth/0.npy",
              "normal": "train/normal/0.npy", "alpha": "train/alpha/0.npy",
              "rgb_gt": "train/rgb_gt/0.png", "metadata": "train/metadata/0.json"}
    result = rebase_scene_samples([sample], "HNI0366")[0]
    assert result["rgb_render"] == "HNI0366/train/rgb_render/0.png"
    assert result["depth"] == "HNI0366/train/depth/0.npy"


def test_scene_arguments_uses_saved_cfg_and_explicit_overrides(tmp_path):
    model = tmp_path / "model"; model.mkdir()
    (model / "cfg_args").write_text(
        "Namespace(sh_degree=3, source_path='old', model_path='old', images='images', depths='', "
        "resolution=2, white_background=False, train_test_exp=False, data_device='cuda', eval=False, "
        "depth_prior_dir='', normal_prior_dir='', confidence_prior_dir='')", encoding="utf-8")
    cli = Namespace(resolution=1, split="train,val")
    merged = scene_arguments(cli, tmp_path / "scene", model)
    assert merged.resolution == 1 and merged.sh_degree == 3
    assert merged.source_path == str((tmp_path / "scene").resolve())
