from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import (
    EVENT_LABEL_TO_RAW,
    EVENT_MARKER_STYLE_RAW,
    IMAGE_SIZE,
    KILL_EVENTS,
    LOOT_EVENTS,
    MAP_CONFIG,
    MINIMAP_PATH,
    MOVEMENT_EVENTS,
    DEATH_EVENTS,
)
from time_utils import format_ts


PLAYER_TYPE_COLORS: dict[str, str] = {
    "Human": "#22c55e",
    "Bot": "#3b82f6",
}

MATCH_COLOR_PALETTE: list[str] = [
    "#22c55e",
    "#3b82f6",
    "#f59e0b",
    "#ef4444",
    "#a855f7",
    "#14b8a6",
    "#f97316",
    "#eab308",
    "#06b6d4",
    "#ec4899",
    "#84cc16",
    "#8b5cf6",
    "#10b981",
    "#f43f5e",
    "#6366f1",
    "#d946ef",
]


if isinstance(IMAGE_SIZE, (tuple, list)) and len(IMAGE_SIZE) == 2:
    IMAGE_WIDTH = int(IMAGE_SIZE[0])
    IMAGE_HEIGHT = int(IMAGE_SIZE[1])
else:
    IMAGE_WIDTH = int(IMAGE_SIZE)
    IMAGE_HEIGHT = int(IMAGE_SIZE)


def safe_listdir(path: Path) -> list[str]:
    try:
        return os.listdir(path)
    except Exception:
        return []


def get_date_folders(data_path: Path) -> list[str]:
    folders: list[str] = []
    for name in safe_listdir(data_path):
        full_path = data_path / name
        if full_path.is_dir() and not name.startswith("."):
            folders.append(name)
    return sorted(folders)


def load_day(folder_path_str: str) -> tuple[pd.DataFrame, int, int]:
    folder_path = Path(folder_path_str)
    frames: list[pd.DataFrame] = []
    loaded_count = 0
    failed_count = 0

    for file_name in safe_listdir(folder_path):
        if file_name.startswith("."):
            continue

        full_path = folder_path / file_name
        if not full_path.is_file():
            continue

        try:
            df_part = pd.read_parquet(full_path)
            frames.append(df_part)
            loaded_count += 1
        except Exception:
            failed_count += 1

    if not frames:
        return pd.DataFrame(), loaded_count, failed_count

    df = pd.concat(frames, ignore_index=True)
    return df, loaded_count, failed_count


def decode_event_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "event" not in out.columns:
        out["event"] = ""
        return out

    out["event"] = out["event"].apply(
        lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x
    )
    out["event"] = out["event"].astype(str)
    return out


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "event" not in out.columns:
        out["event"] = ""
    out["event_norm"] = out["event"].astype(str).str.strip().str.lower()

    if "user_id" not in out.columns:
        out["user_id"] = ""
    out["user_id_str"] = out["user_id"].astype(str).str.strip()

    if "match_id" not in out.columns:
        out["match_id"] = ""
    out["match_id"] = out["match_id"].astype(str).str.strip()

    if "map_id" not in out.columns:
        if "map" in out.columns:
            out["map_id"] = out["map"]
        else:
            out["map_id"] = ""
    out["map_id"] = out["map_id"].astype(str).str.strip()

    out["is_bot"] = out["user_id_str"].str.fullmatch(r"\d+").fillna(False)
    out["player_type"] = np.where(out["is_bot"], "Bot", "Human")

    if "x" not in out.columns:
        out["x"] = np.nan
    if "z" not in out.columns:
        out["z"] = np.nan

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["z"] = pd.to_numeric(out["z"], errors="coerce")

    # Critical for playback:
    # promote whichever raw timestamp column exists into event_time
    if "event_time" not in out.columns:
        if "timestamp" in out.columns:
            out["event_time"] = out["timestamp"]
        elif "ts" in out.columns:
            out["event_time"] = out["ts"]
        elif "time" in out.columns:
            out["event_time"] = out["time"]
        else:
            out["event_time"] = pd.NaT

    def classify_event_group(event_name: str) -> str:
        if event_name in MOVEMENT_EVENTS:
            return "movement"
        if event_name in KILL_EVENTS:
            return "kill"
        if event_name in DEATH_EVENTS:
            return "death"
        if event_name in LOOT_EVENTS:
            return "loot"
        return "other"

    out["event_group"] = out["event_norm"].map(classify_event_group)
    return out


def map_coordinates(df: pd.DataFrame, origin_x: float, origin_z: float, scale: float) -> pd.DataFrame:
    out = df.copy()

    if "x" not in out.columns:
        out["x"] = np.nan
    if "z" not in out.columns:
        out["z"] = np.nan

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["z"] = pd.to_numeric(out["z"], errors="coerce")
    out = out.dropna(subset=["x", "z"]).copy()

    safe_scale = float(scale) if float(scale) != 0 else 1.0

    out["u"] = (out["x"] - float(origin_x)) / safe_scale
    out["v"] = (out["z"] - float(origin_z)) / safe_scale

    out["pixel_x"] = out["u"] * IMAGE_WIDTH
    out["pixel_y"] = (1.0 - out["v"]) * IMAGE_HEIGHT

    out["in_bounds"] = (
        out["pixel_x"].between(0, IMAGE_WIDTH)
        & out["pixel_y"].between(0, IMAGE_HEIGHT)
    )
    return out


def pixel_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total_rows = int(len(df))
    in_bounds = int(df["in_bounds"].sum()) if "in_bounds" in df.columns else 0
    out_of_bounds = total_rows - in_bounds

    return {
        "pixel_x_min": float(df["pixel_x"].min()),
        "pixel_x_max": float(df["pixel_x"].max()),
        "pixel_y_min": float(df["pixel_y"].min()),
        "pixel_y_max": float(df["pixel_y"].max()),
        "in_bounds": in_bounds,
        "out_of_bounds": out_of_bounds,
        "total_rows": total_rows,
    }


def load_minimap(map_name: str) -> Image.Image:
    if map_name not in MAP_CONFIG:
        raise FileNotFoundError(f"Unknown map '{map_name}'. Add it to MAP_CONFIG.")

    minimap_file = MAP_CONFIG[map_name]["image"]
    minimap_path = MINIMAP_PATH / minimap_file

    if not minimap_path.exists():
        raise FileNotFoundError(f"Minimap not found: {minimap_path}")

    img = Image.open(minimap_path).convert("RGBA")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return img


def get_match_color_map(match_ids: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for idx, match_id in enumerate(match_ids):
        color_map[match_id] = MATCH_COLOR_PALETTE[idx % len(MATCH_COLOR_PALETTE)]
    return color_map


def filter_by_player_type(
    df: pd.DataFrame,
    include_human_journeys: bool,
    include_bot_journeys: bool,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    allowed = []
    if include_human_journeys:
        allowed.append("Human")
    if include_bot_journeys:
        allowed.append("Bot")

    if not allowed:
        return df.iloc[0:0].copy()

    return df[df["player_type"].isin(allowed)].copy()


def get_raw_event_for_focus(event_focus: str) -> Optional[str]:
    if event_focus == "All Events":
        return None
    return EVENT_LABEL_TO_RAW.get(event_focus)


def apply_event_focus_filter(df: pd.DataFrame, event_focus: str) -> pd.DataFrame:
    if df.empty or event_focus == "All Events":
        return df.copy()

    raw_event = get_raw_event_for_focus(event_focus)
    if raw_event is None:
        return df.copy()

    return df[df["event_norm"] == raw_event].copy()


def get_focus_highlight_mask(df: pd.DataFrame, event_focus: str) -> pd.Series:
    if df.empty or event_focus == "All Events":
        return pd.Series(False, index=df.index)

    raw_event = get_raw_event_for_focus(event_focus)
    if raw_event is None:
        return pd.Series(False, index=df.index)

    return df["event_norm"] == raw_event


def add_heatmap_trace(fig: go.Figure, df: pd.DataFrame, bins: int = 40) -> None:
    if df.empty:
        return

    x = df["pixel_x"].to_numpy(dtype=float)
    y = df["plot_y"].to_numpy(dtype=float)

    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0, IMAGE_WIDTH], [0, IMAGE_HEIGHT]],
    )

    z = hist.T
    z[z == 0] = np.nan

    if np.isnan(z).all():
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=z,
            hoverinfo="skip",
            showscale=False,
            zsmooth="best",
            opacity=0.6,
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [0.15, "rgba(255,255,0,0.20)"],
                [0.45, "rgba(255,165,0,0.45)"],
                [0.75, "rgba(255,69,0,0.65)"],
                [1.0, "rgba(255,0,0,0.85)"],
            ],
        )
    )


def build_hotspot_summary(
    df: pd.DataFrame,
    cell_size: int,
    label_prefix: str = "Zone",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    hotspot_df = df.dropna(subset=["pixel_x", "pixel_y"]).copy()
    if hotspot_df.empty:
        return pd.DataFrame()

    hotspot_df["cell_x"] = (hotspot_df["pixel_x"] // cell_size).astype(int)
    hotspot_df["cell_y"] = (hotspot_df["pixel_y"] // cell_size).astype(int)

    summary = (
        hotspot_df.groupby(["cell_x", "cell_y"])
        .agg(
            event_count=("event_norm", "size"),
            matches=("match_id", lambda s: s.astype(str).nunique()),
            unique_players=("user_id_str", "nunique"),
            human_journeys=("player_type", lambda s: int((s == "Human").sum())),
            bot_journeys=("player_type", lambda s: int((s == "Bot").sum())),
            event_types=("event_norm", lambda s: ", ".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
        .sort_values(
            ["event_count", "matches", "unique_players"],
            ascending=[False, False, False],
        )
    )

    summary["zone"] = (
        label_prefix + " (" +
        summary["cell_x"].astype(str) + ", " +
        summary["cell_y"].astype(str) + ")"
    )

    cols = [
        "zone",
        "event_count",
        "matches",
        "unique_players",
        "human_journeys",
        "bot_journeys",
        "event_types",
        "cell_x",
        "cell_y",
    ]
    return summary[cols]


def build_focused_match_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.assign(match_id_str=df["match_id"].astype(str))
        .groupby("match_id_str")
        .agg(
            event_rows=("match_id_str", "size"),
            unique_players=("user_id_str", "nunique"),
            human_journeys=("player_type", lambda s: int((s == "Human").sum())),
            bot_journeys=("player_type", lambda s: int((s == "Bot").sum())),
            first_event=("event_time", "min"),
            last_event=("event_time", "max"),
        )
        .reset_index()
        .rename(columns={"match_id_str": "match_id"})
        .sort_values(["event_rows", "match_id"], ascending=[False, True])
    )

    if not out.empty:
        out["first_event"] = out["first_event"].apply(format_ts)
        out["last_event"] = out["last_event"].apply(format_ts)

    return out


def _apply_fixed_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        range=[0, IMAGE_WIDTH],
        showgrid=False,
        zeroline=False,
        visible=False,
        fixedrange=False,
    )
    fig.update_yaxes(
        range=[0, IMAGE_HEIGHT],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=False,
    )
    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="pan",
        plot_bgcolor="black",
        paper_bgcolor="black",
        uirevision="keep-view",
    )


def build_plotly_figure(
    base_map: Image.Image,
    df: pd.DataFrame,
    max_players_per_match: int,
    show_lines: bool,
    show_event_markers: bool,
    show_heatmap: bool,
    color_by_match: bool,
    focused_event_label: Optional[str],
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, IMAGE_WIDTH],
            y=[0, IMAGE_HEIGHT],
            mode="markers",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_layout_image(
        dict(
            source=base_map,
            x=0,
            y=IMAGE_HEIGHT,
            sizex=IMAGE_WIDTH,
            sizey=IMAGE_HEIGHT,
            xref="x",
            yref="y",
            layer="below",
            sizing="stretch",
            opacity=1.0,
        )
    )

    if df.empty:
        _apply_fixed_axes(fig)
        return fig

    render_df = df.copy()
    render_df["plot_y"] = IMAGE_HEIGHT - render_df["pixel_y"]

    if show_heatmap:
        add_heatmap_trace(fig, render_df)

    selected_match_ids = sorted(render_df["match_id"].astype(str).unique().tolist())
    match_color_map = get_match_color_map(selected_match_ids)

    movement_df = render_df[render_df["event_group"] == "movement"].copy()

    if show_lines and not movement_df.empty:
        movement_df["match_id_str"] = movement_df["match_id"].astype(str)

        for match_id, match_part in movement_df.groupby("match_id_str", sort=False):
            top_players = (
                match_part.groupby("user_id_str")
                .size()
                .sort_values(ascending=False)
                .head(max_players_per_match)
                .index.tolist()
            )

            match_part = match_part[match_part["user_id_str"].isin(top_players)].copy()
            match_part = match_part.sort_values(["user_id_str", "event_time", "event_norm"])

            for _, player_df in match_part.groupby("user_id_str", sort=False):
                if player_df.empty:
                    continue

                player_type = player_df["player_type"].iloc[0]

                line_color = (
                    match_color_map.get(str(match_id), "#FFFFFF")
                    if color_by_match
                    else PLAYER_TYPE_COLORS.get(player_type, "#FFFFFF")
                )

                line_dash = "solid" if player_type == "Human" else "dot"
                endpoint_symbol = "circle" if player_type == "Human" else "triangle-up"

                fig.add_trace(
                    go.Scatter(
                        x=player_df["pixel_x"],
                        y=player_df["plot_y"],
                        mode="lines",
                        line=dict(color=line_color, width=2, dash=line_dash),
                        opacity=0.9,
                        showlegend=False,
                        hovertemplate=(
                            "Match: %{customdata[0]}<br>"
                            "Journey: %{customdata[1]}<br>"
                            "Journey Type: %{customdata[2]}<br>"
                            "Event: %{customdata[3]}<br>"
                            "Time: %{customdata[4]}<extra></extra>"
                        ),
                        customdata=player_df[
                            ["match_id", "user_id_str", "player_type", "event_norm", "event_time"]
                        ].values,
                    )
                )

                last_row = player_df.iloc[-1]
                fig.add_trace(
                    go.Scatter(
                        x=[last_row["pixel_x"]],
                        y=[last_row["plot_y"]],
                        mode="markers",
                        marker=dict(
                            size=7,
                            symbol=endpoint_symbol,
                            color=line_color,
                            line=dict(color="white", width=1),
                        ),
                        showlegend=False,
                        hovertemplate=(
                            "Match: %{customdata[0]}<br>"
                            "Journey: %{customdata[1]}<br>"
                            "Journey Type: %{customdata[2]}<br>"
                            "Latest visible movement point<extra></extra>"
                        ),
                        customdata=np.array([
                            [
                                str(last_row["match_id"]),
                                str(last_row["user_id_str"]),
                                str(last_row["player_type"]),
                            ]
                        ]),
                    )
                )

    if show_event_markers:
        event_rows = render_df[render_df["event_norm"].isin(EVENT_MARKER_STYLE_RAW.keys())].copy()

        if not event_rows.empty:
            if color_by_match:
                event_rows["match_id_str"] = event_rows["match_id"].astype(str)

                for _, row in event_rows.iterrows():
                    marker_cfg = EVENT_MARKER_STYLE_RAW.get(
                        row["event_norm"],
                        {"symbol": "circle", "color": "#FFFFFF", "size": 8},
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[row["pixel_x"]],
                            y=[row["plot_y"]],
                            mode="markers",
                            marker=dict(
                                symbol=marker_cfg["symbol"],
                                size=marker_cfg["size"],
                                color=match_color_map.get(str(row["match_id_str"]), "#FFFFFF"),
                                line=dict(color="white", width=1),
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "Match: %{customdata[0]}<br>"
                                "Journey: %{customdata[1]}<br>"
                                "Journey Type: %{customdata[2]}<br>"
                                "Event: %{customdata[3]}<br>"
                                "Time: %{customdata[4]}<extra></extra>"
                            ),
                            customdata=np.array([
                                [
                                    str(row["match_id"]),
                                    str(row["user_id_str"]),
                                    str(row["player_type"]),
                                    str(row["event_norm"]),
                                    str(row["event_time"]),
                                ]
                            ]),
                        )
                    )
            else:
                for event_name, marker_cfg in EVENT_MARKER_STYLE_RAW.items():
                    event_df = event_rows[event_rows["event_norm"] == event_name].copy()
                    if event_df.empty:
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x=event_df["pixel_x"],
                            y=event_df["plot_y"],
                            mode="markers",
                            marker=dict(
                                symbol=marker_cfg["symbol"],
                                size=marker_cfg["size"],
                                color=marker_cfg["color"],
                                line=dict(color="white", width=1),
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "Match: %{customdata[0]}<br>"
                                "Journey: %{customdata[1]}<br>"
                                "Journey Type: %{customdata[2]}<br>"
                                "Event: %{customdata[3]}<br>"
                                "Time: %{customdata[4]}<extra></extra>"
                            ),
                            customdata=event_df[
                                ["match_id", "user_id_str", "player_type", "event_norm", "event_time"]
                            ].values,
                        )
                    )

    if focused_event_label:
        focus_mask = get_focus_highlight_mask(render_df, focused_event_label)
        focus_df = render_df[focus_mask].copy()

        if not focus_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=focus_df["pixel_x"],
                    y=focus_df["plot_y"],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=16,
                        color="white",
                        line=dict(color="#00FFFF", width=2),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "Focused Event<br>"
                        "Match: %{customdata[0]}<br>"
                        "Journey: %{customdata[1]}<br>"
                        "Journey Type: %{customdata[2]}<br>"
                        "Event: %{customdata[3]}<br>"
                        "Time: %{customdata[4]}<extra></extra>"
                    ),
                    customdata=focus_df[
                        ["match_id", "user_id_str", "player_type", "event_norm", "event_time"]
                    ].values,
                )
            )

    _apply_fixed_axes(fig)
    return fig


def build_match_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.assign(match_id_str=df["match_id"].astype(str))
        .groupby("match_id_str")
        .agg(
            rows=("match_id_str", "size"),
            journeys=("user_id_str", "nunique"),
            kill_side_events=("event_group", lambda s: int((s == "kill").sum())),
            death_side_events=("event_group", lambda s: int((s == "death").sum())),
            loot=("event_group", lambda s: int((s == "loot").sum())),
            movement=("event_group", lambda s: int((s == "movement").sum())),
            first_seen=("event_time", "min"),
            last_seen=("event_time", "max"),
        )
        .reset_index()
        .rename(columns={"match_id_str": "match_id"})
        .sort_values(["rows", "match_id"], ascending=[False, True])
    )

    if not summary.empty:
        summary["first_seen"] = summary["first_seen"].apply(format_ts)
        summary["last_seen"] = summary["last_seen"].apply(format_ts)

    return summary


def build_event_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cols = [
        "event_time",
        "match_id",
        "user_id_str",
        "player_type",
        "event_norm",
        "event_group",
        "x",
        "z",
        "pixel_x",
        "pixel_y",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    out = df[existing_cols].copy()

    if "event_time" in out.columns:
        out["event_time"] = out["event_time"].apply(format_ts)

    sort_cols = [c for c in ["event_time", "match_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=True, na_position="last")

    return out


def show_onboarding_card(title: str, body: str, bullets: list[str]) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 22px;
            background: rgba(255,255,255,0.03);
            margin-top: 10px;
            margin-bottom: 10px;
        ">
            <h3 style="margin-top:0;">{title}</h3>
            <p style="margin-bottom:12px;">{body}</p>
            <ul style="margin-bottom:0;">
                {''.join([f'<li>{item}</li>' for item in bullets])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.02);
            min-height: 96px;
        ">
            <div style="
                font-size: 0.95rem;
                color: rgba(255,255,255,0.75);
                margin-bottom: 8px;
                line-height: 1.2;
                word-break: break-word;
            ">
                {label}
            </div>
            <div style="
                font-size: 2.1rem;
                font-weight: 700;
                line-height: 1.1;
                white-space: normal;
                word-break: break-word;
                overflow-wrap: anywhere;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_scope_label(selected_matches_active: bool, event_focus: str, match_filter_mode: str) -> str:
    if selected_matches_active and match_filter_mode == "search_filter" and event_focus != "All Events":
        return f"Search-filtered matches focused on {event_focus}"
    if selected_matches_active and match_filter_mode == "search_filter":
        return "Search-filtered matches"
    if selected_matches_active and match_filter_mode == "explicit_selection" and event_focus != "All Events":
        return f"Selected matches focused on {event_focus}"
    if selected_matches_active and match_filter_mode == "explicit_selection":
        return "Selected matches"
    if event_focus != "All Events":
        return f"Full day + map focused on {event_focus}"
    return "Full day + map"