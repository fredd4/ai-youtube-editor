"""Stable segment identity: anchors survive renumbering, QC catches a pickup over speech."""
from __future__ import annotations

import json
from pathlib import Path

from ytedit.project import Project
from ytedit.qc import voice_pickup_overlap_issues
from ytedit.timeline import Timeline, VideoSegment, VoiceAnchor, VoiceItem

from tests.test_tidy import SPACED, add_clip, seg, timeline_of


def _anchored(tl: Timeline, voice_id: str, segment: VideoSegment, offset: float = 0.0) -> VoiceItem:
    item = VoiceItem(
        id=voice_id, file="voice/n001.wav", at=0.0, end=2.0,
        anchor=VoiceAnchor(segment=segment.id, offset=offset, uid=segment.uid),
    )
    tl.tracks.voice.append(item)
    tl.resolve_anchors()
    return item


def test_uids_are_assigned_persisted_and_stable(tmp_path: Path) -> None:
    tl = timeline_of(seg("s001", "c001", 0.0, 4.0), seg("s002", "c001", 10.0, 14.0))
    uids = [s.uid for s in tl.tracks.video]
    assert all(uids) and len(set(uids)) == 2
    path = tmp_path / "timeline.json"
    tl.save(path)
    assert [s.uid for s in Timeline.load(path).tracks.video] == uids
    # a file written before uids existed gets deterministic ones on load
    raw = json.loads(path.read_text())
    for s in raw["tracks"]["video"]:
        s.pop("uid", None)
    path.write_text(json.dumps(raw))
    first = [s.uid for s in Timeline.load(path).tracks.video]
    second = [s.uid for s in Timeline.load(path).tracks.video]
    assert first == second and all(first)


def test_anchor_follows_its_segment_through_renumbering() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 4.0),
        seg("s002", "c001", 10.0, 14.0, mute_source=True, role="b-roll"),
    )
    target = tl.tracks.video[1]
    item = _anchored(tl, "v001", target)
    assert item.at == tl.segment_positions()[1].start
    # three new segments in front, then the display ids are renumbered —
    # the anchor must still resolve to the same picture, not to "s002"
    tl.insert_segments(0, [seg("sX", "c001", 20.0, 22.0) for _ in range(3)])
    for i, s in enumerate(tl.tracks.video, 1):
        s.id = f"s{i:03d}"
    tl.resolve_anchors()
    pos = next(p for p in tl.segment_positions() if p.segment.uid == target.uid)
    assert item.at == pos.start
    assert item.anchor is not None and item.anchor.segment == target.id


def test_lost_uid_is_repaired_from_signature() -> None:
    tl = timeline_of(seg("s001", "c001", 0.0, 4.0), seg("s002", "c001", 10.0, 14.0))
    target = tl.tracks.video[1]
    item = _anchored(tl, "v001", target)
    assert item.anchor is not None and item.anchor.signature is not None
    item.anchor.uid = "deadbeef"  # stale
    tl.insert_segments(0, [seg("sX", "c001", 20.0, 22.0)])
    tl.resolve_anchors()
    pos = next(p for p in tl.segment_positions() if p.segment.uid == target.uid)
    assert item.at == pos.start
    assert not tl.meta.anchor_issues


def test_rule_33_flags_a_pickup_over_on_camera_speech(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)  # transcript words across the clip
    tl = timeline_of(
        seg("s001", "c001", 3.0, 8.0),                                   # narration on camera
        seg("s002", "c001", 20.0, 26.0, mute_source=True, role="b-roll"),  # muted picture
    )
    over_speech = VoiceItem(id="v001", file="voice/n001.wav", at=1.0, end=4.0)
    over_muted = VoiceItem(id="v002", file="voice/n002.wav", at=5.5, end=8.5)
    tl.tracks.voice = [over_speech, over_muted]
    issues = voice_pickup_overlap_issues(project, tl)
    assert any("v001" in msg for msg in issues)
    assert not any("v002" in msg for msg in issues)
