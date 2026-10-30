import os
import shutil

import pytest

from sfs.data.manifest import (assign_protocol_b, assign_validation, build_manifest,
                               check_leakage, gate_warnings, index_videos, main, split_from_path)

GROUP = {"train": "kitware-meva-training", "eval": "kitware"}


def _make_tree(tmp_path, act_path, clips):
    """clips: [(stem, split, has_video)] -> (anno_root, video_root)."""
    anno = tmp_path / "repo"
    vids = tmp_path / "videos"
    vids.mkdir()
    for stem, split, has_video in clips:
        d = anno / "annotation" / "DIVA-phase-2" / "MEVA" / GROUP[split] / stem[:10]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(act_path, d / f"{stem}.activities.yml")
        if has_video:
            (vids / f"{stem}.r13.avi").write_bytes(b"")
    return str(anno), str(vids)


def _stem(camera, minute="05"):
    return f"2018-03-13.17-{minute}-00.17-10-00.school.{camera}"


def test_split_from_path():
    assert split_from_path("/r/annotation/DIVA-phase-2/MEVA/kitware-meva-training/a.yml") == "train"
    assert split_from_path("/r/annotation/DIVA-phase-2/MEVA/kitware/a.yml") == "eval"
    assert split_from_path("/r/elsewhere/a.yml") == "other"


def test_build_keeps_only_clips_with_video(tmp_path, act_path):
    anno, vids = _make_tree(tmp_path, act_path, [(_stem("G1"), "train", True),
                                                 (_stem("G2"), "eval", True),
                                                 (_stem("G3"), "train", False)])
    df, stats = build_manifest(anno, index_videos(vids))
    assert stats["instances_in_annotations"] == 15  # 3 files x 5 activities
    assert len(df) == 10 and set(df.camera) == {"G1", "G2"}
    assert set(df.split) == {"train", "eval"}
    assert list(df.instance_id) == list(range(10))


def test_duplicates_are_dropped(tmp_path, act_path):
    anno, vids = _make_tree(tmp_path, act_path, [(_stem("G1"), "train", True)])
    extra = os.path.join(anno, "annotation", "DIVA-phase-2", "MEVA", "kitware-meva-training", "dup")
    os.makedirs(extra)
    shutil.copy(act_path, os.path.join(extra, f"{_stem('G1')}.activities.yml"))
    df, stats = build_manifest(anno, index_videos(vids))
    assert len(df) == 5 and stats["duplicates_dropped"] == 5


def test_leakage_raises(tmp_path, act_path):
    anno, vids = _make_tree(tmp_path, act_path, [(_stem("G1"), "train", True)])
    d = os.path.join(anno, "annotation", "DIVA-phase-2", "MEVA", "kitware", "x")
    os.makedirs(d)
    p = os.path.join(d, f"{_stem('G1')}.activities.yml")
    with open(act_path) as src, open(p, "w") as dst:  # same clip, different activity window
        dst.write(src.read().replace("[7999, 8088]", "[1, 50]"))
    df, _ = build_manifest(anno, index_videos(vids))
    with pytest.raises(ValueError, match="both train and eval"):
        check_leakage(df)


def test_validation_is_held_out_by_clip(tmp_path, act_path):
    clips = [(_stem(f"G{i}", f"{i:02d}"), "train", True) for i in range(10)] + [(_stem("E1"), "eval", True)]
    anno, vids = _make_tree(tmp_path, act_path, clips)
    df, _ = build_manifest(anno, index_videos(vids))
    df = assign_validation(df, frac=0.2, seed=1)
    val, train = set(df[df.split == "val"].clip_stem), set(df[df.split == "train"].clip_stem)
    assert len(val) == 2 and not (val & train)
    assert set(df[df.split == "eval"].clip_stem) == {_stem("E1")}  # eval untouched
    assert assign_validation(df, 0.2, 1).equals(assign_validation(df, 0.2, 1))  # deterministic


def test_protocol_b_holds_out_whole_cameras(tmp_path, act_path):
    clips = [(_stem(f"G{i}", f"{i:02d}"), "train", True) for i in range(10)]
    anno, vids = _make_tree(tmp_path, act_path, clips)
    df, _ = build_manifest(anno, index_videos(vids))
    df = assign_protocol_b(df, eval_frac=0.3, val_frac=0.1, seed=0)
    eval_cams = set(df[df.split_b == "eval"].camera)
    train_cams = set(df[df.split_b.isin(["train", "val"])].camera)
    assert eval_cams and not (eval_cams & train_cams)
    assert (df.split_b == "eval").mean() >= 0.3


def test_gate_warns_on_thin_eval(tmp_path, act_path):
    anno, vids = _make_tree(tmp_path, act_path, [(_stem("E1"), "eval", True)])
    df, _ = build_manifest(anno, index_videos(vids))
    warns = gate_warnings(df)
    assert any("eval has 5 instances" in w for w in warns)


def test_cli_end_to_end(tmp_path, act_path, capsys):
    anno, vids = _make_tree(tmp_path, act_path, [(_stem("G1"), "train", True),
                                                 (_stem("G2", "06"), "train", True),
                                                 (_stem("E1"), "eval", True)])
    out = str(tmp_path / "m" / "manifest.csv")
    assert main(["--anno", anno, "--videos", vids, "--out", out]) == 0
    assert os.path.exists(out) and "manifest written" in capsys.readouterr().out
