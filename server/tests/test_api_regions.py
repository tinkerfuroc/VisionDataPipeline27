from test_api_labeling import _client, _clip
from vdp.engine.fake import FakeEngine


def test_put_regions_round_trips_sorted(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        body = {"regions": [
            {"start": 10, "end": 19, "crop": [0, 0, 60, 50], "blocks": [[10, 10, 20, 20]]},
            {"start": 0, "end": 4, "crop": None, "blocks": [[0, 0, 5, 5]]},
        ]}
        r = client.put(f"/api/clips/{cid}/regions", json=body)
        assert r.status_code == 200, r.text
        assert [g["start"] for g in r.json()["clip"]["regions"]] == [0, 10]
        got = client.get(f"/api/clips/{cid}").json()["clip"]["regions"]
        assert got[1] == {"start": 10, "end": 19, "crop": [0, 0, 60, 50], "blocks": [[10, 10, 20, 20]]}
        assert client.put(f"/api/clips/{cid}/regions", json={"regions": []}).json()["clip"]["regions"] == []


def test_put_regions_errors(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        bad = client.put(f"/api/clips/{cid}/regions", json={"regions": [
            {"start": 0, "end": 5, "blocks": []}, {"start": 5, "end": 6, "blocks": []}]})
        assert bad.status_code == 400 and "overlaps" in bad.json()["detail"]
        assert client.put("/api/clips/nope/regions", json={"regions": []}).status_code == 404
        assert client.put(f"/api/clips/{cid}/regions", json={"regions": [{"start": 0, "end": 1, "blocks": [[1, 2]]}]}).status_code == 422


def test_prompt_route_refuses_an_excluded_click(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.put(f"/api/clips/{cid}/regions", json={"regions": [{"start": 0, "end": 19, "blocks": [[20, 20, 40, 40]]}]})
        r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 5, "prompt": {"type": "clicks", "points": [[30, 30, 1]]}, "object_id": None})
        assert r.status_code == 400 and r.json()["detail"] == "that spot is excluded on this frame"
