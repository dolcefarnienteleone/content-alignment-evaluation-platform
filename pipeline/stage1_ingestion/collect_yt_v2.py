from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import isodate
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "selected_videos.csv"
OUTPUT_DIR = Path("data/raw")

MAX_TOP_LEVEL_COMMENTS = 100
INCLUDE_ALL_REPLIES = True

# "relevance" is useful for V1 content-alignment analysis.
# Later, V2 Alignment Drift may additionally collect order="time".
COMMENT_ORDER = "relevance"

TRANSCRIPT_LANGUAGES = ["ko", "en"]

# Small pause between videos; not strictly required but keeps ingestion polite.
SLEEP_BETWEEN_VIDEOS_SECONDS = 1


# ============================================================
# SETUP
# ============================================================

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "YOUTUBE_API_KEY is missing. Add it to your .env file."
    )

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_video_id(url_or_id: str) -> str:
    """
    Extract a YouTube video ID from common URL formats.

    Supports:
    - raw 11-character ID
    - youtube.com/watch?v=...
    - youtu.be/...
    - youtube.com/shorts/...
    - youtube.com/embed/...
    """

    value = str(url_or_id).strip()

    if re.fullmatch(r"[\w-]{11}", value):
        return value

    parsed = urlparse(value)

    hostname = (parsed.hostname or "").lower()

    if hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]

    elif hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]

        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/")[1].split("/")[0]

        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/")[1].split("/")[0]

        else:
            video_id = ""

    else:
        video_id = ""

    if not re.fullmatch(r"[\w-]{11}", video_id):
        raise ValueError(
            f"Could not extract a valid YouTube video ID from: {url_or_id}"
        )

    return video_id


def extract_hashtags(
    title: str,
    description: str,
) -> list[str]:
    """
    Extract hashtags from title + description.
    """

    text = f"{title}\n{description}"

    hashtags = re.findall(
        r"(?<!\w)#([\w가-힣ぁ-んァ-ヶ一-龥]+)",
        text,
    )

    seen = set()
    output = []

    for tag in hashtags:
        key = tag.casefold()

        if key not in seen:
            seen.add(key)
            output.append(tag)

    return output


def anonymize_author(
    channel_id: str | None,
    display_name: str | None,
) -> str | None:
    """
    Produce a stable anonymized audience identifier.
    """

    raw_value = channel_id or display_name

    if not raw_value:
        return None

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:20]


def calculate_hours_after_publish(
    comment_time: str | None,
    publish_time: str | None,
) -> float | None:
    """
    Calculate comment time relative to content publish time.
    """

    if not comment_time or not publish_time:
        return None

    try:
        comment_dt = pd.to_datetime(
            comment_time,
            utc=True,
        )

        publish_dt = pd.to_datetime(
            publish_time,
            utc=True,
        )

        hours = (
            comment_dt - publish_dt
        ).total_seconds() / 3600

        return round(hours, 3)

    except Exception:
        return None


def assign_engagement_stage(
    hours_after_publish: float | None,
) -> str | None:
    """
    V1 stage logic.
    This will support Alignment Drift later.
    """

    if hours_after_publish is None:
        return None

    if hours_after_publish < 0:
        return "invalid"

    if hours_after_publish <= 24:
        return "early"

    if hours_after_publish <= 24 * 7:
        return "growing"

    if hours_after_publish <= 24 * 30:
        return "sustained"

    return "long_tail"


# ============================================================
# VIDEO METADATA
# ============================================================

def get_video_metadata(
    video_id: str,
    case_name: str | None = None,
) -> dict[str, Any]:

    response = (
        youtube
        .videos()
        .list(
            part="snippet,contentDetails,statistics",
            id=video_id,
        )
        .execute()
    )

    if not response.get("items"):
        raise ValueError(
            f"Video not found or unavailable: {video_id}"
        )

    item = response["items"][0]

    snippet = item["snippet"]
    details = item["contentDetails"]
    statistics = item.get("statistics", {})

    title = snippet.get("title", "")
    description = snippet.get("description", "")

    duration = isodate.parse_duration(
        details["duration"]
    )

    content_id = f"youtube_{video_id}"

    return {
        "content_id": content_id,
        "case_name": case_name,
        "platform": "youtube",
        "platform_content_id": video_id,
        "creator_id": snippet.get("channelId"),
        "creator_name": snippet.get("channelTitle"),
        "content_type": "video",
        "title": title,
        "description": description,

        # JSON strings are easier to preserve safely in CSV.
        "hashtags": json.dumps(
            extract_hashtags(
                title,
                description,
            ),
            ensure_ascii=False,
        ),

        "youtube_tags": json.dumps(
            snippet.get("tags", []),
            ensure_ascii=False,
        ),

        "publish_time": snippet.get("publishedAt"),

        "url": (
            f"https://www.youtube.com/watch?v={video_id}"
        ),

        "duration_seconds": int(
            duration.total_seconds()
        ),

        "language": (
            snippet.get("defaultAudioLanguage")
            or snippet.get("defaultLanguage")
        ),

        "view_count": int(
            statistics.get("viewCount", 0)
        ),

        "like_count": int(
            statistics.get("likeCount", 0)
        ),

        "comment_count": int(
            statistics.get("commentCount", 0)
        ),

        "collected_at": utc_now_iso(),
    }


# ============================================================
# TRANSCRIPT
# ============================================================

def get_transcript(
    video_id: str,
    content_id: str,
    preferred_languages: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    preferred_languages = (
        preferred_languages
        or ["ko", "en"]
    )

    try:
        api = YouTubeTranscriptApi()

        transcript_list = api.list(video_id)

        transcript = (
            transcript_list
            .find_transcript(
                preferred_languages
            )
        )

        fetched = transcript.fetch()

        rows = []

        for i, segment in enumerate(fetched):

            rows.append(
                {
                    "transcript_segment_id":
                        f"{content_id}_seg_{i:05d}",

                    "content_id":
                        content_id,

                    "platform_content_id":
                        video_id,

                    "segment_index":
                        i,

                    "text":
                        segment.text,

                    "start_seconds":
                        segment.start,

                    "duration_seconds":
                        segment.duration,

                    "end_seconds":
                        segment.start
                        + segment.duration,

                    "language":
                        fetched.language,

                    "language_code":
                        fetched.language_code,

                    "is_generated":
                        fetched.is_generated,

                    "collected_at":
                        utc_now_iso(),
                }
            )

        status = {
            "transcript_status":
                "success",

            "transcript_language":
                fetched.language,

            "transcript_language_code":
                fetched.language_code,

            "transcript_is_generated":
                fetched.is_generated,

            "transcript_segment_count":
                len(rows),
        }

        return (
            pd.DataFrame(rows),
            status,
        )

    except Exception as exc:

        status = {
            "transcript_status":
                "failed",

            "transcript_error_type":
                type(exc).__name__,

            "transcript_error":
                str(exc),

            "transcript_segment_count":
                0,
        }

        return (
            pd.DataFrame(),
            status,
        )


# ============================================================
# COMMENTS
# ============================================================

def parse_comment(
    item: dict[str, Any],
    *,
    content_id: str,
    video_id: str,
    publish_time: str,
    parent_comment_id: str | None,
) -> dict[str, Any]:

    snippet = item["snippet"]

    author_channel = (
        snippet.get(
            "authorChannelId",
            {},
        )
    )

    author_channel_id = (
        author_channel.get("value")
    )

    comment_time = (
        snippet.get("publishedAt")
    )

    hours_after_publish = (
        calculate_hours_after_publish(
            comment_time,
            publish_time,
        )
    )

    return {
        "comment_id":
            f"youtube_{item['id']}",

        "content_id":
            content_id,

        "platform":
            "youtube",

        "platform_content_id":
            video_id,

        "platform_comment_id":
            item["id"],

        "parent_comment_id":
            (
                f"youtube_{parent_comment_id}"
                if parent_comment_id
                else None
            ),

        "author_id_hash":
            anonymize_author(
                author_channel_id,
                snippet.get(
                    "authorDisplayName"
                ),
            ),

        "comment_text":
            snippet.get(
                "textDisplay",
                "",
            ),

        "comment_time":
            comment_time,

        "updated_time":
            snippet.get(
                "updatedAt"
            ),

        "like_count":
            int(
                snippet.get(
                    "likeCount",
                    0,
                )
            ),

        "hours_after_publish":
            hours_after_publish,

        "engagement_stage":
            assign_engagement_stage(
                hours_after_publish
            ),

        "is_reply":
            parent_comment_id
            is not None,

        "is_creator_reply":
            False,

        "collection_strategy":
            COMMENT_ORDER,

        "collected_at":
            utc_now_iso(),
    }


def get_all_replies(
    parent_comment_id: str,
    *,
    content_id: str,
    video_id: str,
    publish_time: str,
) -> list[dict[str, Any]]:

    replies = []

    page_token = None

    while True:

        response = (
            youtube
            .comments()
            .list(
                part="snippet",
                parentId=parent_comment_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText",
            )
            .execute()
        )

        for item in response.get(
            "items",
            [],
        ):

            replies.append(
                parse_comment(
                    item,
                    content_id=content_id,
                    video_id=video_id,
                    publish_time=publish_time,
                    parent_comment_id=parent_comment_id,
                )
            )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return replies


def get_comments(
    video_id: str,
    *,
    content_id: str,
    publish_time: str,
    max_top_level_comments: int = 100,
    include_all_replies: bool = True,
    order: str = "relevance",
) -> pd.DataFrame:

    comments = []

    page_token = None

    top_level_count = 0

    try:

        while (
            top_level_count
            < max_top_level_comments
        ):

            page_size = min(
                100,
                max_top_level_comments
                - top_level_count,
            )

            response = (
                youtube
                .commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=page_size,
                    pageToken=page_token,
                    textFormat="plainText",
                    order=order,
                )
                .execute()
            )

            for thread in response.get(
                "items",
                [],
            ):

                top_comment = (
                    thread[
                        "snippet"
                    ][
                        "topLevelComment"
                    ]
                )

                comments.append(
                    parse_comment(
                        top_comment,
                        content_id=content_id,
                        video_id=video_id,
                        publish_time=publish_time,
                        parent_comment_id=None,
                    )
                )

                top_level_count += 1

                total_reply_count = int(
                    thread[
                        "snippet"
                    ].get(
                        "totalReplyCount",
                        0,
                    )
                )

                if (
                    include_all_replies
                    and total_reply_count > 0
                ):

                    comments.extend(
                        get_all_replies(
                            top_comment["id"],
                            content_id=content_id,
                            video_id=video_id,
                            publish_time=publish_time,
                        )
                    )

            page_token = response.get(
                "nextPageToken"
            )

            if not page_token:
                break

    except HttpError as exc:

        if exc.resp.status == 403:

            print(
                f"[WARNING] Comments unavailable "
                f"for video {video_id}."
            )

            return pd.DataFrame()

        raise

    return pd.DataFrame(comments)


# ============================================================
# SINGLE VIDEO PIPELINE
# ============================================================

def collect_single_video(
    url_or_id: str,
    *,
    case_name: str | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    video_id = extract_video_id(
        url_or_id
    )

    print(
        f"[INFO] Collecting video: "
        f"{video_id}"
    )

    metadata = get_video_metadata(
        video_id,
        case_name=case_name,
    )

    content_id = metadata[
        "content_id"
    ]

    publish_time = metadata[
        "publish_time"
    ]

    transcript_df, transcript_status = (
        get_transcript(
            video_id,
            content_id,
            preferred_languages=
                TRANSCRIPT_LANGUAGES,
        )
    )

    metadata.update(
        transcript_status
    )

    comments_df = get_comments(
        video_id,
        content_id=content_id,
        publish_time=publish_time,
        max_top_level_comments=
            MAX_TOP_LEVEL_COMMENTS,
        include_all_replies=
            INCLUDE_ALL_REPLIES,
        order=COMMENT_ORDER,
    )

    metadata["collected_comment_rows"] = (
        len(comments_df)
    )

    metadata_df = pd.DataFrame(
        [metadata]
    )

    print(
        f"[INFO] Completed {video_id}: "
        f"{len(transcript_df)} transcript segments, "
        f"{len(comments_df)} comment/reply rows."
    )

    return (
        metadata_df,
        transcript_df,
        comments_df,
    )


# ============================================================
# BATCH PIPELINE
# ============================================================

def load_selected_videos(
    csv_path: str,
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    required_columns = {
        "video_url",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "selected_videos.csv is missing "
            f"required columns: {sorted(missing)}"
        )

    if "case_name" not in df.columns:
        df["case_name"] = None

    df = df[
        ["video_url", "case_name"]
    ].copy()

    df["video_url"] = (
        df["video_url"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["video_url"] != ""
    ]

    df = (
        df
        .drop_duplicates(
            subset=["video_url"]
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid videos found "
            "in selected_videos.csv."
        )

    return df


def run_batch_collection(
    input_csv: str,
) -> None:

    selected_df = load_selected_videos(
        input_csv
    )

    print(
        f"[INFO] Loaded "
        f"{len(selected_df)} selected videos."
    )

    all_content = []
    all_transcripts = []
    all_comments = []

    run_log = []

    for row_num, row in (
        selected_df.iterrows()
    ):

        video_url = row["video_url"]

        case_name = (
            row["case_name"]
            if pd.notna(
                row["case_name"]
            )
            else None
        )

        try:

            (
                metadata_df,
                transcript_df,
                comments_df,
            ) = collect_single_video(
                video_url,
                case_name=case_name,
            )

            all_content.append(
                metadata_df
            )

            if not transcript_df.empty:
                all_transcripts.append(
                    transcript_df
                )

            if not comments_df.empty:
                all_comments.append(
                    comments_df
                )

            run_log.append(
                {
                    "video_url":
                        video_url,

                    "case_name":
                        case_name,

                    "status":
                        "success",

                    "content_id":
                        metadata_df.iloc[
                            0
                        ][
                            "content_id"
                        ],

                    "transcript_status":
                        metadata_df.iloc[
                            0
                        ].get(
                            "transcript_status"
                        ),

                    "transcript_rows":
                        len(
                            transcript_df
                        ),

                    "comment_rows":
                        len(
                            comments_df
                        ),

                    "error_type":
                        None,

                    "error_message":
                        None,

                    "processed_at":
                        utc_now_iso(),
                }
            )

        except Exception as exc:

            print(
                f"[ERROR] Failed: "
                f"{video_url}"
            )

            print(
                f"        "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            run_log.append(
                {
                    "video_url":
                        video_url,

                    "case_name":
                        case_name,

                    "status":
                        "failed",

                    "content_id":
                        None,

                    "transcript_status":
                        None,

                    "transcript_rows":
                        0,

                    "comment_rows":
                        0,

                    "error_type":
                        type(exc).__name__,

                    "error_message":
                        str(exc),

                    "processed_at":
                        utc_now_iso(),
                }
            )

        if (
            SLEEP_BETWEEN_VIDEOS_SECONDS
            > 0
        ):
            time.sleep(
                SLEEP_BETWEEN_VIDEOS_SECONDS
            )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    if all_content:

        content_df = pd.concat(
            all_content,
            ignore_index=True,
        )

    else:

        content_df = pd.DataFrame()

    if all_transcripts:

        transcript_df = pd.concat(
            all_transcripts,
            ignore_index=True,
        )

    else:

        transcript_df = pd.DataFrame()

    if all_comments:

        comments_df = pd.concat(
            all_comments,
            ignore_index=True,
        )

    else:

        comments_df = pd.DataFrame()

    run_log_df = pd.DataFrame(
        run_log
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if (
        not content_df.empty
        and content_df[
            "content_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate content_id detected "
            "in content output."
        )

    if (
        not comments_df.empty
        and comments_df[
            "comment_id"
        ].duplicated().any()
    ):

        duplicate_count = (
            comments_df[
                "comment_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            f"[WARNING] "
            f"{duplicate_count} duplicate "
            "comment IDs detected. "
            "Keeping first occurrence."
        )

        comments_df = (
            comments_df
            .drop_duplicates(
                subset=[
                    "comment_id"
                ],
                keep="first",
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    content_path = (
        OUTPUT_DIR
        / "content.csv"
    )

    transcript_path = (
        OUTPUT_DIR
        / "transcript_segment.csv"
    )

    comments_path = (
        OUTPUT_DIR
        / "audience_comment.csv"
    )

    log_path = (
        OUTPUT_DIR
        / "youtube_ingestion_run_log.csv"
    )

    content_df.to_csv(
        content_path,
        index=False,
    )

    transcript_df.to_csv(
        transcript_path,
        index=False,
    )

    comments_df.to_csv(
        comments_path,
        index=False,
    )

    run_log_df.to_csv(
        log_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    success_count = (
        run_log_df[
            "status"
        ]
        .eq("success")
        .sum()
    )

    failed_count = (
        run_log_df[
            "status"
        ]
        .eq("failed")
        .sum()
    )

    print()
    print(
        "======================================"
    )
    print(
        "YouTube Batch Ingestion Complete"
    )
    print(
        "======================================"
    )

    print(
        f"Selected videos : "
        f"{len(selected_df)}"
    )

    print(
        f"Successful      : "
        f"{success_count}"
    )

    print(
        f"Failed          : "
        f"{failed_count}"
    )

    print(
        f"Content rows    : "
        f"{len(content_df)}"
    )

    print(
        f"Transcript rows : "
        f"{len(transcript_df)}"
    )

    print(
        f"Comment rows    : "
        f"{len(comments_df)}"
    )

    print()
    print(
        f"Saved: {content_path}"
    )

    print(
        f"Saved: {transcript_path}"
    )

    print(
        f"Saved: {comments_path}"
    )

    print(
        f"Saved: {log_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_batch_collection(
        INPUT_CSV
    )