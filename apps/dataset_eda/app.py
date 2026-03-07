import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, cast

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "raw" / "videos.json"


def _to_abs_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _iter_audio_refs(obj: Any) -> Iterable[str]:
    audio_exts = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

    if isinstance(obj, str):
        s = obj.strip()
        lower = s.lower()
        if lower.startswith("http://") or lower.startswith("https://"):
            if any(lower.endswith(ext) for ext in audio_exts):
                yield s
            return
        if any(lower.endswith(ext) for ext in audio_exts):
            yield s
        return

    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_audio_refs(v)
        return

    if isinstance(obj, list):
        for v in obj:
            yield from _iter_audio_refs(v)
        return


def _mime_for_audio_path(p: Path) -> str | None:
    ext = p.suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".m4a":
        return "audio/mp4"
    return None


def _discover_tracks(video_id: str) -> list[Path]:
    if not video_id:
        return []

    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    tracks: list[Path] = []
    for split in ("train", "test"):
        d = REPO_ROOT / "data" / "raw" / split / video_id
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in audio_exts:
                tracks.append(p)
    return tracks


def _get_query_record() -> int | None:
    if hasattr(st, "query_params"):
        v = st.query_params.get("record")
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    qp = st.experimental_get_query_params()
    raw = qp.get("record")
    if not raw:
        return None
    try:
        return int(raw[0])
    except Exception:
        return None


def _set_query_record(record: int | None) -> None:
    if hasattr(st, "query_params"):
        if record is None:
            st.query_params.clear()
        else:
            st.query_params["record"] = str(record)
        st.rerun()
        return

    if record is None:
        st.experimental_set_query_params()
    else:
        st.experimental_set_query_params(record=str(record))
    st.experimental_rerun()
    return


def _render_details(records: list[dict[str, Any]], record_idx: int) -> None:
    if record_idx < 0 or record_idx >= len(records):
        st.error(f"Record index out of range: {record_idx}")
        if st.button("Back to list"):
            _set_query_record(None)
        st.stop()

    r = records[int(record_idx)]

    if st.button("Back to list"):
        _set_query_record(None)

    title = str(r.get("title", ""))
    video_id = str(r.get("videoId", ""))
    published_at = str(r.get("publishedAt", ""))
    position = r.get("position")
    description = str(r.get("description", ""))
    poly_any = r.get("polyphony")
    poly = cast(dict[str, Any], poly_any) if isinstance(poly_any, dict) else {}
    poly_url = str(poly.get("url", ""))
    poly_code = str(poly.get("code", ""))

    st.subheader(title or video_id or "(record)")
    cols = st.columns(3)
    cols[0].metric("Video ID", video_id or "-")
    cols[1].metric("Published", published_at or "-")
    cols[2].metric("Position", str(position) if position is not None else "-")

    link_cols = st.columns(2)
    if video_id:
        link_cols[0].link_button(
            "Open on YouTube", f"https://www.youtube.com/watch?v={video_id}"
        )
    if poly_url:
        link_cols[1].link_button(f"Open Polyphony ({poly_code or 'link'})", poly_url)

    if description:
        st.text_area("Description", value=description, height=240)

    st.divider()
    st.markdown("### Audio")

    discovered_tracks = _discover_tracks(video_id)
    if discovered_tracks:
        st.caption("Discovered track files in data/raw/(train|test)/<videoId>:")
        for p in discovered_tracks:
            st.write(
                str(p.relative_to(REPO_ROOT))
                if str(p).startswith(str(REPO_ROOT))
                else str(p)
            )
            st.audio(p.read_bytes(), format=_mime_for_audio_path(p))
    else:
        st.caption("No track files discovered under data/raw/(train|test)/<videoId>.")

    audio_refs: list[str] = []
    audio_refs.extend(_iter_audio_refs(r))
    local_mp3_path = r.get("local_mp3_path")
    if isinstance(local_mp3_path, str) and local_mp3_path.strip():
        audio_refs.append(local_mp3_path.strip())

    seen: set[str] = set()
    audio_refs = [x for x in audio_refs if not (x in seen or seen.add(x))]

    if audio_refs:
        st.markdown("#### Other audio references")
        for ref in audio_refs:
            if ref.startswith("http://") or ref.startswith("https://"):
                st.write(ref)
                st.audio(ref)
                continue

            p = _to_abs_path(ref)
            if not p.exists():
                st.warning(f"Missing audio file: {ref}")
                continue

            st.write(
                str(p.relative_to(REPO_ROOT))
                if str(p).startswith(str(REPO_ROOT))
                else str(p)
            )
            st.audio(p.read_bytes(), format=_mime_for_audio_path(p))

    st.divider()
    st.markdown("### Record (raw)")
    st.json(r)


def _render_list(records: list[dict[str, Any]], search_index: list[str]) -> None:
    st.subheader("Browse records")

    q = (
        st.text_input(
            "Search",
            value="",
            placeholder="title / videoId / code",
            key="_dataset_eda_search",
        )
        .strip()
        .lower()
    )
    cols = st.columns([0.25, 0.25, 0.5])
    page_size = int(
        cols[0].selectbox(
            "Page size",
            options=[10, 25, 50, 100, 200],
            index=2,
            key="_dataset_eda_page_size",
        )
    )

    if q:
        matched = [i for i, s in enumerate(search_index) if q in s]
    else:
        matched = list(range(len(records)))

    if not matched:
        st.warning("No records match your search.")
        st.stop()

    page_count = max(1, math.ceil(len(matched) / page_size))

    if "_dataset_eda_page" not in st.session_state:
        st.session_state["_dataset_eda_page"] = 1

    last_q = st.session_state.get("_dataset_eda_last_q")
    last_page_size = st.session_state.get("_dataset_eda_last_page_size")
    if last_q != q or last_page_size != page_size:
        st.session_state["_dataset_eda_page"] = 1
    st.session_state["_dataset_eda_last_q"] = q
    st.session_state["_dataset_eda_last_page_size"] = page_size

    if int(st.session_state["_dataset_eda_page"]) > page_count:
        st.session_state["_dataset_eda_page"] = page_count

    page = int(
        cols[1].number_input(
            "Page",
            min_value=1,
            max_value=page_count,
            value=int(st.session_state["_dataset_eda_page"]),
            step=1,
            key="_dataset_eda_page",
        )
    )
    cols[2].caption(f"Showing {len(matched)} matches")

    start = (page - 1) * page_size
    end = start + page_size
    page_idxs = matched[start:end]

    nav = st.columns([0.15, 0.15, 0.7])
    if nav[0].button("Prev", disabled=page <= 1):
        st.session_state["_dataset_eda_page"] = max(1, page - 1)
        st.rerun()
    if nav[1].button("Next", disabled=page >= page_count):
        st.session_state["_dataset_eda_page"] = min(page_count, page + 1)
        st.rerun()

    st.divider()

    for idx in page_idxs:
        r = records[idx]
        title = str(r.get("title", "(no title)"))
        video_id = str(r.get("videoId", ""))
        poly_code = ""
        poly = r.get("polyphony")
        if isinstance(poly, dict):
            poly_code = str(poly.get("code", ""))

        title_btn = title
        if len(title_btn) > 100:
            title_btn = title_btn[:97] + "..."

        row = st.columns([0.18, 0.18, 0.64])
        row[0].caption(video_id or "-")
        row[1].caption(poly_code or "-")
        if row[2].button(title_btn, key=f"open_{idx}", use_container_width=True):
            _set_query_record(idx)


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str) -> tuple[list[dict[str, Any]], list[str], str]:
    p = _to_abs_path(dataset_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        data = data["videos"]

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list, got: {type(data).__name__}")

    records: list[dict[str, Any]] = []
    search_index: list[str] = []
    for i, rec in enumerate(data):
        if isinstance(rec, dict):
            records.append(rec)
        else:
            records.append({"_index": i, "_value": rec})

        r = records[-1]
        title = str(r.get("title", ""))
        video_id = str(r.get("videoId", ""))
        poly = r.get("polyphony")
        poly_code = str(poly.get("code", "")) if isinstance(poly, dict) else ""
        search_index.append(f"{title} {video_id} {poly_code}".lower())

    return records, search_index, str(p)


def main() -> None:
    st.set_page_config(page_title="Dataset EDA (videos.json)", layout="wide")
    st.title("Dataset EDA: videos.json")

    with st.sidebar:
        st.header("Browse")

        default_dataset_path = os.getenv("DATASET_PATH") or str(
            DEFAULT_DATASET.relative_to(REPO_ROOT)
        )
        default_dataset_path = default_dataset_path.strip()
        if default_dataset_path.startswith("@"):
            default_dataset_path = default_dataset_path[1:]

        dataset_path = st.text_input(
            "Dataset path",
            value=default_dataset_path,
            help="Default: data/raw/videos.json (repo-relative)",
        )
        dataset_path = dataset_path.strip()
        if dataset_path.startswith("@"):
            dataset_path = dataset_path[1:]
        if st.button("Reload dataset"):
            st.cache_data.clear()

    try:
        records, search_index, resolved_path = load_dataset(dataset_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()
        return

    with st.sidebar:
        st.caption(f"Loaded: {resolved_path}")

    record_idx = _get_query_record()
    if record_idx is None:
        _render_list(records, search_index)
    else:
        _render_details(records, record_idx)


if __name__ == "__main__":
    main()
