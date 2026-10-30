import pytest

from sfs.data.kpf import clip_stem, geom_path_for, parse_activities, parse_geom, parse_stem
from tests.conftest import STEM


def test_parse_activities_real_file(act_path):
    acts = parse_activities(act_path)
    assert len(acts) == 5
    stop = next(a for a in acts if a.activity == "Vehicle_Stopping")
    assert (stop.frame_start, stop.frame_end, stop.actor_ids, stop.act_id) == (7999, 8088, (4,), 1)


def test_parse_geom_real_file(geom_path):
    tracks = parse_geom(geom_path)
    assert tracks[0][3710] == (1071, 698, 1173, 771)


def test_every_actor_has_boxes_inside_its_activity(act_path, geom_path):
    tracks = parse_geom(geom_path)
    for a in parse_activities(act_path):
        for actor in a.actor_ids:
            frames = [f for f in tracks[actor] if a.frame_start <= f <= a.frame_end]
            assert frames, f"{a.activity}: actor {actor} has no boxes in its window"


def test_clip_stem_strips_known_suffixes():
    for suffix in (".activities.yml", ".geom.yml", ".r13.avi", ".avi", ".mp4"):
        assert clip_stem(f"/x/y/{STEM}{suffix}") == STEM


def test_parse_stem():
    k = parse_stem(STEM)
    assert (k.date, k.start, k.site, k.camera, k.hour) == ("2018-03-13", "17-05-00", "school", "G328", "17")
    with pytest.raises(ValueError):
        parse_stem("not-a-stem")


def test_geom_path_for():
    assert geom_path_for(f"/a/{STEM}.activities.yml") == f"/a/{STEM}.geom.yml"
    with pytest.raises(ValueError):
        geom_path_for("/a/x.geom.yml")


def test_skips_malformed_entries(tmp_path):
    p = tmp_path / "x.activities.yml"
    p.write_text(
        '- { meta: "header" }\n'
        "- { act: { act2: { Talk: 1.0 }, id2: 3, timespan: [{tsr0: [10, 20]}], actors: [{ id1: 7 }] } }\n"
        "- { act: { act2: { NoSpan: 1.0 }, id2: 4 } }\n"
        "- { act: { id2: 5, timespan: [{tsr0: [1, 2]}] } }\n"
    )
    acts = parse_activities(str(p))
    assert [(a.activity, a.actor_ids) for a in acts] == [("Talk", (7,))]


# Regression: the MEVA repo mixes YAML styles. The first Colab run parsed 0 tracks for
# 250/252 clips because a regex only matched one style. Lines below are copied verbatim.
def test_parse_geom_quoted_key_style():
    from tests.conftest import FIXTURES
    import os
    tracks = parse_geom(os.path.join(FIXTURES, "geom_quoted_style.geom.yml"))
    assert tracks[1][2677] == (1182, 267, 1326, 540) and len(tracks[1]) == 4


def test_parse_geom_big_ids_and_meta_lines():
    from tests.conftest import FIXTURES
    import os
    tracks = parse_geom(os.path.join(FIXTURES, "geom_bigid_style.geom.yml"))
    assert tracks[6068717397083399654][2478] == (487, 902, 967, 1080)


def test_parse_geom_float_coordinates(tmp_path):
    p = tmp_path / "f.geom.yml"
    p.write_text("- { geom: {id1: 2, ts0: 10, g0: 10.4 20.6 30.0 40.5} }\n")
    assert parse_geom(str(p))[2][10] == (10, 21, 30, 40)
