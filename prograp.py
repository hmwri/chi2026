#!/usr/bin/env python3
"""Download the public CHI 2026 SIGCHI program JSON and export CSV(s).

This script relies on the public program endpoint used by SIGCHI's conference
program app. It exports:

1) chi2026_program_all_tracks.csv  -> all content rows with matched sessions
2) chi2026_poster_demo_papers.csv  -> filtered to papers/talks, posters, demos

Notes:
- Japanese translation is intentionally left blank because this script is meant
  to create the canonical complete English/session CSV first.
- Encoding is UTF-8 with BOM for Excel compatibility.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import Request, urlopen

PROGRAM_URL = "https://files.sigchi.org/conference/program/CHI/2026"
OUT_DIR = Path(".")
RAW_JSON = OUT_DIR / "chi2026_program_raw.json"
ALL_CSV = OUT_DIR / "chi2026_program_all_tracks.csv"
FILTERED_CSV = OUT_DIR / "chi2026_poster_demo_papers.csv"


def fetch_json(url: str) -> Dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; chi2026-csv-export/1.0)",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=60) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        text = resp.read().decode(charset)
    return json.loads(text)


def safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(normalize_text(v) for v in value if v is not None)
    return str(value).strip()


def normalize_rich_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace(r"\&", "&")
    return re.sub(r"\s+", " ", text).strip()


def join_unique(values: Iterable[Any]) -> str:
    seen = set()
    out = []
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return " | ".join(out)


def json_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_dt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        if isinstance(value, (int, float)):
            # SIGCHI sample app compares startDate numerically, so timestamps are plausible.
            # Treat large values as milliseconds since epoch; smaller as seconds.
            ts = float(value)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return (
                dt.datetime.fromtimestamp(ts, dt.UTC)
                .isoformat(timespec="minutes")
                .replace("+00:00", "Z")
            )
        if isinstance(value, str):
            # Pass through ISO-like strings after mild normalization.
            s = value.strip()
            if not s:
                return ""
            # Try parse common ISO variants for consistency.
            for candidate in (s.replace("Z", "+00:00"), s):
                try:
                    parsed = dt.datetime.fromisoformat(candidate)
                    if parsed.tzinfo is None:
                        return parsed.isoformat(timespec="minutes")
                    return parsed.isoformat(timespec="minutes")
                except ValueError:
                    pass
            return s
    except Exception:
        return str(value)
    return str(value)


def pick_room(session: Dict[str, Any]) -> str:
    direct = safe_get(
        session,
        "roomName",
        "room",
        "locationName",
        "location",
        "venue",
        "place",
        "space",
        "meetingRoom",
        default="",
    )
    if direct:
        return normalize_text(direct)
    for key in ("rooms", "locations", "venues"):
        if key in session and session[key]:
            return normalize_text(session[key])
    for key in ("roomIds", "locationIds", "venueIds"):
        if key in session and session[key]:
            return normalize_text(session[key])
    return ""


def pick_content_type_id(content: Dict[str, Any]) -> Any:
    return safe_get(content, "contentTypeId", "typeId", "kindId", default=None)


def pick_title(content: Dict[str, Any]) -> str:
    return normalize_text(
        safe_get(content, "title", "name", "contentTitle", "headline", default="")
    )


def pick_abstract(content: Dict[str, Any]) -> str:
    return normalize_rich_text(
        safe_get(content, "abstract", "description", "summary", default="")
    )


def format_person_name(person: Dict[str, Any]) -> str:
    parts = [
        normalize_text(person.get("firstName")),
        normalize_text(person.get("middleInitial")),
        normalize_text(person.get("lastName")),
    ]
    return " ".join(part for part in parts if part)


def pick_author_metadata(
    content: Dict[str, Any], people_by_id: Dict[Any, Dict[str, Any]]
) -> Dict[str, str]:
    ids = []
    names = []
    affiliations = []
    departments = []
    cities = []
    states = []
    countries = []
    for author in content.get("authors", []) or []:
        if not isinstance(author, dict):
            continue
        person_id = author.get("personId")
        ids.append(person_id)
        person = people_by_id.get(person_id, {})
        name = format_person_name(person)
        if name:
            names.append(name)
        for affiliation in author.get("affiliations", []) or person.get("affiliations", []) or []:
            if not isinstance(affiliation, dict):
                continue
            institution = normalize_text(affiliation.get("institution"))
            dsl = normalize_text(affiliation.get("dsl"))
            city = normalize_text(affiliation.get("city"))
            state = normalize_text(affiliation.get("state"))
            country = normalize_text(affiliation.get("country"))
            if institution and dsl:
                affiliations.append(f"{institution} ({dsl})")
            elif institution:
                affiliations.append(institution)
            if dsl:
                departments.append(dsl)
            if city:
                cities.append(city)
            if state:
                states.append(state)
            if country:
                countries.append(country)
    return {
        "author_ids": join_unique(ids),
        "authors": join_unique(names),
        "affiliations": join_unique(affiliations),
        "author_departments": join_unique(departments),
        "author_cities": join_unique(cities),
        "author_states": join_unique(states),
        "author_countries": join_unique(countries),
        "authors_json": json_text(content.get("authors")),
    }


def pick_people_names(ids: Iterable[Any], people_by_id: Dict[Any, Dict[str, Any]]) -> str:
    return join_unique(format_person_name(people_by_id.get(person_id, {})) for person_id in ids)


def pick_keywords(content: Dict[str, Any]) -> str:
    values = []
    for keyword in content.get("keywords", []) or []:
        if isinstance(keyword, dict):
            values.append(safe_get(keyword, "name", "title", "label", default=""))
        else:
            values.append(keyword)
    return join_unique(values)


def pick_doi(content: Dict[str, Any]) -> str:
    doi = (content.get("addons") or {}).get("doi", {})
    if not isinstance(doi, dict):
        return ""
    url = normalize_text(doi.get("url"))
    if url.startswith("https://doi.org/"):
        return url.removeprefix("https://doi.org/")
    return url


def pick_addon_url(content: Dict[str, Any], addon_name: str) -> str:
    addon = (content.get("addons") or {}).get(addon_name, {})
    if not isinstance(addon, dict):
        return ""
    return normalize_text(addon.get("url"))


def pick_addon_urls(content: Dict[str, Any]) -> str:
    addons = content.get("addons") or {}
    if not isinstance(addons, dict):
        return ""
    return join_unique(
        addon.get("url")
        for addon in addons.values()
        if isinstance(addon, dict) and addon.get("url")
    )


def build_floor_by_room_id(floors: Iterable[Dict[str, Any]]) -> Dict[Any, str]:
    floor_by_room_id = {}
    for floor in floors:
        if not isinstance(floor, dict):
            continue
        name = normalize_text(floor.get("name"))
        for room_id in floor.get("roomIds", []) or []:
            if name:
                floor_by_room_id[room_id] = name
    return floor_by_room_id


def pick_url(content_id: Any) -> str:
    return f"https://programs.sigchi.org/chi/2026/program/content/{content_id}"


def pick_session_url(session_id: Any) -> str:
    return f"https://programs.sigchi.org/chi/2026/program/session/{session_id}"


def classify_track(content_type_name: str, session_name: str = "") -> str:
    text = f"{content_type_name} {session_name}".lower()
    if "demo" in text:
        return "demo"
    if "poster" in text:
        return "poster"
    if "paper" in text or "talk" in text:
        return "paper_or_talk"
    return "other"


def iter_rows(data: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    conference = data.get("conference", {})
    publication_info = data.get("publicationInfo", {})
    sessions: List[Dict[str, Any]] = data.get("sessions", []) or []
    contents: List[Dict[str, Any]] = data.get("contents", []) or []
    content_types: List[Dict[str, Any]] = data.get("contentTypes", []) or []
    time_slots: List[Dict[str, Any]] = data.get("timeSlots", []) or []
    people: List[Dict[str, Any]] = data.get("people", []) or []
    tracks: List[Dict[str, Any]] = data.get("tracks", []) or []
    rooms: List[Dict[str, Any]] = data.get("rooms", []) or []
    events: List[Dict[str, Any]] = data.get("events", []) or []
    recognitions: List[Dict[str, Any]] = data.get("recognitions", []) or []
    floors: List[Dict[str, Any]] = data.get("floors", []) or []

    content_type_by_id: Dict[Any, Dict[str, Any]] = {
        ct.get("id"): ct for ct in content_types if isinstance(ct, dict)
    }
    time_slot_by_id: Dict[Any, Dict[str, Any]] = {
        ts.get("id"): ts for ts in time_slots if isinstance(ts, dict)
    }
    people_by_id: Dict[Any, Dict[str, Any]] = {
        person.get("id"): person for person in people if isinstance(person, dict)
    }
    track_by_id: Dict[Any, Dict[str, Any]] = {
        track.get("id"): track for track in tracks if isinstance(track, dict)
    }
    room_by_id: Dict[Any, Dict[str, Any]] = {
        room.get("id"): room for room in rooms if isinstance(room, dict)
    }
    event_by_id: Dict[Any, Dict[str, Any]] = {
        event.get("id"): event for event in events if isinstance(event, dict)
    }
    recognition_by_id: Dict[Any, Dict[str, Any]] = {
        recognition.get("id"): recognition
        for recognition in recognitions
        if isinstance(recognition, dict)
    }
    floor_by_room_id = build_floor_by_room_id(floors)

    sessions_by_content_id: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        for cid in session.get("contentIds", []) or []:
            sessions_by_content_id[cid].append(session)

    events_by_content_id: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        for cid in event.get("contentIds", []) or []:
            events_by_content_id[cid].append(event)

    base_conf_name = normalize_text(safe_get(conference, "title", "name", default="CHI 2026"))
    base_conf_short = normalize_text(safe_get(conference, "shortName", default="chi"))
    base_year = normalize_text(safe_get(conference, "year", default="2026"))
    base_conf_start = normalize_dt(conference.get("startDate"))
    base_conf_end = normalize_dt(conference.get("endDate"))

    for content in contents:
        content_id = content.get("id")
        title_en = pick_title(content)
        content_type_id = pick_content_type_id(content)
        ct_obj = content_type_by_id.get(content_type_id, {})
        content_type_name = normalize_text(safe_get(ct_obj, "name", "title", default=""))
        author_metadata = pick_author_metadata(content, people_by_id)
        track = track_by_id.get(content.get("trackId"), {})
        content_events = [
            event_by_id.get(event_id, {})
            for event_id in content.get("eventIds", []) or []
            if event_by_id.get(event_id)
        ] or events_by_content_id.get(content_id, [])
        recognition_names = [
            safe_get(recognition_by_id.get(recognition_id, {}), "name", "title", default="")
            for recognition_id in content.get("recognitionIds", []) or []
        ]
        event_rooms = [
            safe_get(room_by_id.get(event.get("roomId"), {}), "name", default="")
            for event in content_events
        ]
        event_room_ids = [
            event.get("roomId") for event in content_events if isinstance(event, dict)
        ]
        event_floors = [floor_by_room_id.get(room_id, "") for room_id in event_room_ids]

        matched_sessions = sessions_by_content_id.get(content_id) or [None]
        for session in matched_sessions:
            session_name = normalize_text(session.get("name") if session else "")
            time_slot = time_slot_by_id.get(session.get("timeSlotId")) if session else None
            start = normalize_dt(time_slot.get("startDate") if time_slot else "")
            end = normalize_dt(time_slot.get("endDate") if time_slot else "")
            session_type = content_type_by_id.get(session.get("typeId") if session else None, {})
            session_room = room_by_id.get(session.get("roomId") if session else None, {})
            row = {
                "conference_id": normalize_text(conference.get("id")),
                "conference": base_conf_name,
                "conference_short": base_conf_short,
                "conference_display_short": normalize_text(conference.get("displayShortName")),
                "conference_year": base_year,
                "conference_full_name": normalize_text(conference.get("fullName")),
                "conference_url": normalize_text(conference.get("url")),
                "conference_location": normalize_text(conference.get("location")),
                "conference_start": base_conf_start,
                "conference_end": base_conf_end,
                "conference_timezone_name": normalize_text(conference.get("timeZoneName")),
                "conference_timezone_offset": normalize_text(conference.get("timeZoneOffset")),
                "publication_status": normalize_text(publication_info.get("publicationStatus")),
                "publication_date": normalize_text(publication_info.get("publicationDate")),
                "publication_version": normalize_text(publication_info.get("version")),
                "track_group": classify_track(content_type_name, session_name),
                "track_id": normalize_text(content.get("trackId")),
                "track_name": normalize_text(track.get("name")),
                "track_type_id": normalize_text(track.get("typeId")),
                "content_type_id": normalize_text(content_type_id),
                "content_type": content_type_name,
                "content_type_display_name": normalize_text(ct_obj.get("displayName")),
                "content_type_color": normalize_text(ct_obj.get("color")),
                "content_duration_minutes": normalize_text(ct_obj.get("duration")),
                "content_duration_override": normalize_text(content.get("durationOverride")),
                "content_imported_id": normalize_text(content.get("importedId")),
                "content_source": normalize_text(content.get("source")),
                "is_break": normalize_text(content.get("isBreak")),
                "award": normalize_text(content.get("award")),
                "tags": join_unique(content.get("tags", []) or []),
                "session_name": session_name,
                "session_description": normalize_rich_text(session.get("description") if session else ""),
                "session_imported_id": normalize_text(session.get("importedId") if session else ""),
                "session_source": normalize_text(session.get("source") if session else ""),
                "session_type_id": normalize_text(session.get("typeId") if session else ""),
                "session_type": normalize_text(safe_get(session_type, "name", "title", default="")),
                "session_is_parallel_presentation": normalize_text(
                    session.get("isParallelPresentation") if session else ""
                ),
                "session_chair_ids": join_unique(session.get("chairIds", []) if session else []),
                "session_chairs": pick_people_names(
                    session.get("chairIds", []) if session else [], people_by_id
                ),
                "session_content_ids": join_unique(session.get("contentIds", []) if session else []),
                "session_addons_json": json_text(session.get("addons") if session else {}),
                "session_start": start,
                "session_end": end,
                "session_room_id": normalize_text(session.get("roomId") if session else ""),
                "session_room": normalize_text(session_room.get("name")) or pick_room(session) if session else "",
                "session_floor": floor_by_room_id.get(session.get("roomId") if session else None, ""),
                "session_room_type_id": normalize_text(session_room.get("typeId")),
                "session_room_setup": normalize_text(session_room.get("setup")),
                "session_room_capacity": normalize_text(session_room.get("capacity")),
                "session_room_note": normalize_text(session_room.get("note")),
                "title_en": title_en,
                "title_ja": "",
                "abstract_en": pick_abstract(content),
                "abstract_ja": "",
                "author_ids": author_metadata["author_ids"],
                "authors": author_metadata["authors"],
                "affiliations": author_metadata["affiliations"],
                "author_departments": author_metadata["author_departments"],
                "author_cities": author_metadata["author_cities"],
                "author_states": author_metadata["author_states"],
                "author_countries": author_metadata["author_countries"],
                "authors_json": author_metadata["authors_json"],
                "keywords": pick_keywords(content),
                "doi": pick_doi(content),
                "doi_url": pick_addon_url(content, "doi"),
                "presentation_video_url": pick_addon_url(content, "Presentation Video"),
                "addon_urls": pick_addon_urls(content),
                "addons_json": json_text(content.get("addons")),
                "recognition_ids": join_unique(content.get("recognitionIds", []) or []),
                "recognitions": join_unique(recognition_names),
                "content_session_ids": join_unique(content.get("sessionIds", []) or []),
                "content_event_ids": join_unique(
                    [event.get("id") for event in content_events if isinstance(event, dict)]
                ),
                "content_events": join_unique(
                    [event.get("name") for event in content_events if isinstance(event, dict)]
                ),
                "event_start": join_unique(
                    normalize_dt(event.get("startDate"))
                    for event in content_events
                    if isinstance(event, dict)
                ),
                "event_end": join_unique(
                    normalize_dt(event.get("endDate"))
                    for event in content_events
                    if isinstance(event, dict)
                ),
                "event_rooms": join_unique(event_rooms),
                "event_room_ids": join_unique(event_room_ids),
                "event_floors": join_unique(event_floors),
                "event_imported_ids": join_unique(
                    [event.get("importedId") for event in content_events if isinstance(event, dict)]
                ),
                "event_sources": join_unique(
                    [event.get("source") for event in content_events if isinstance(event, dict)]
                ),
                "event_descriptions": join_unique(
                    normalize_rich_text(event.get("description"))
                    for event in content_events
                    if isinstance(event, dict)
                ),
                "event_is_parallel_presentation": join_unique(
                    [event.get("isParallelPresentation") for event in content_events if isinstance(event, dict)]
                ),
                "event_presenter_ids": join_unique(
                    person_id
                    for event in content_events
                    if isinstance(event, dict)
                    for person_id in event.get("presenterIds", []) or []
                ),
                "event_presenters": join_unique(
                    pick_people_names(event.get("presenterIds", []), people_by_id)
                    for event in content_events
                    if isinstance(event, dict)
                ),
                "event_chair_ids": join_unique(
                    person_id
                    for event in content_events
                    if isinstance(event, dict)
                    for person_id in event.get("chairIds", []) or []
                ),
                "event_chairs": join_unique(
                    pick_people_names(event.get("chairIds", []), people_by_id)
                    for event in content_events
                    if isinstance(event, dict)
                ),
                "content_id": normalize_text(content_id),
                "session_id": normalize_text(session.get("id") if session else ""),
                "content_url": pick_url(content_id),
                "session_url": pick_session_url(session.get("id")) if session else "",
            }
            yield row


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "conference_id",
        "conference",
        "conference_short",
        "conference_display_short",
        "conference_year",
        "conference_full_name",
        "conference_url",
        "conference_location",
        "conference_start",
        "conference_end",
        "conference_timezone_name",
        "conference_timezone_offset",
        "publication_status",
        "publication_date",
        "publication_version",
        "track_group",
        "track_id",
        "track_name",
        "track_type_id",
        "content_type_id",
        "content_type",
        "content_type_display_name",
        "content_type_color",
        "content_duration_minutes",
        "content_duration_override",
        "content_imported_id",
        "content_source",
        "is_break",
        "award",
        "tags",
        "session_name",
        "session_description",
        "session_imported_id",
        "session_source",
        "session_type_id",
        "session_type",
        "session_is_parallel_presentation",
        "session_chair_ids",
        "session_chairs",
        "session_content_ids",
        "session_addons_json",
        "session_start",
        "session_end",
        "session_room_id",
        "session_room",
        "session_floor",
        "session_room_type_id",
        "session_room_setup",
        "session_room_capacity",
        "session_room_note",
        "title_en",
        "title_ja",
        "abstract_en",
        "abstract_ja",
        "author_ids",
        "authors",
        "affiliations",
        "author_departments",
        "author_cities",
        "author_states",
        "author_countries",
        "authors_json",
        "keywords",
        "doi",
        "doi_url",
        "presentation_video_url",
        "addon_urls",
        "addons_json",
        "recognition_ids",
        "recognitions",
        "content_session_ids",
        "content_event_ids",
        "content_events",
        "event_start",
        "event_end",
        "event_rooms",
        "event_room_ids",
        "event_floors",
        "event_imported_ids",
        "event_sources",
        "event_descriptions",
        "event_is_parallel_presentation",
        "event_presenter_ids",
        "event_presenters",
        "event_chair_ids",
        "event_chairs",
        "content_id",
        "session_id",
        "content_url",
        "session_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        data = fetch_json(PROGRAM_URL)
    except Exception as exc:
        print(f"Failed to download {PROGRAM_URL}: {exc}", file=sys.stderr)
        return 1

    RAW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = list(iter_rows(data))
    rows.sort(key=lambda r: (r["session_start"], r["session_name"], r["title_en"]))
    write_csv(ALL_CSV, rows)

    filtered = [r for r in rows if r["track_group"] in {"paper_or_talk", "poster", "demo"}]
    write_csv(FILTERED_CSV, filtered)

    print(f"Wrote {RAW_JSON}")
    print(f"Wrote {ALL_CSV} ({len(rows)} rows)")
    print(f"Wrote {FILTERED_CSV} ({len(filtered)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
