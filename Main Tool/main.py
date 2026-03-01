from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    DATA_PATH,
    DATE_PLACEHOLDER,
    FOCUS_EVENT_OPTIONS,
    MAP_CONFIG,
    MAP_PLACEHOLDER,
    MINIMAP_PATH,
)
from data_utils import (
    add_normalized_columns,
    apply_event_focus_filter,
    build_event_table,
    build_focused_match_table,
    build_hotspot_summary,
    build_match_summary,
    build_plotly_figure,
    decode_event_column,
    filter_by_player_type,
    get_date_folders,
    get_scope_label,
    load_day,
    load_minimap,
    map_coordinates,
    pixel_stats,
    render_metric_card,
    show_onboarding_card,
)
from time_utils import apply_timeline_filter, format_ts, parse_timestamp_column

st.set_page_config(
    page_title="LILA Player Journey Explorer",
    layout="wide",
)
st.title("🗺️ LILA Player Journey Explorer")
st.caption(
    "Single-day spatial analysis for player journeys, combat events, loot, and match-level playback."
)


def main() -> None:
    if not DATA_PATH.is_dir():
        st.error(f"Data folder not found: {DATA_PATH}")
        return

    if not MINIMAP_PATH.is_dir():
        st.error(f"Minimap folder not found: {MINIMAP_PATH}")
        return

    date_folders = get_date_folders(DATA_PATH)
    if not date_folders:
        st.error("No date folders found inside player_data.")
        return

    # Sidebar: Core filters
    st.sidebar.header("Filters")

    date_options = [DATE_PLACEHOLDER] + date_folders
    selected_date = st.sidebar.selectbox("Date", date_options, index=0)

    if selected_date == DATE_PLACEHOLDER:
        show_onboarding_card(
            title="Start Here",
            body="Pick a date, then a map. After that, narrow the same map using match filters or a focused event.",
            bullets=[
                "Select a Date",
                "Select a Map",
                "Optionally type or choose Match IDs",
                "Optionally choose an Event Investigation filter",
            ],
        )
        return

    folder_path = DATA_PATH / selected_date

    with st.spinner("Loading parquet data..."):
        df, loaded_count, failed_count = load_day(str(folder_path))

    if df.empty:
        st.error("No parquet data could be loaded.")
        st.caption(f"Files loaded: {loaded_count}, failed: {failed_count}")
        return

    df = decode_event_column(df)
    df = add_normalized_columns(df)
    df = parse_timestamp_column(df)

    required_cols = {"map_id", "match_id", "event_norm", "x", "z", "user_id_str"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return

    available_maps = sorted([m for m in df["map_id"].dropna().unique().tolist() if m in MAP_CONFIG])
    if not available_maps:
        st.error("No supported maps found in selected date.")
        return

    selected_map = st.sidebar.selectbox("Map", [MAP_PLACEHOLDER] + available_maps, index=0)

    if selected_map == MAP_PLACEHOLDER:
        show_onboarding_card(
            title="Pick a Map",
            body=f"Date {selected_date} is loaded. Choose a map to begin analysis.",
            bullets=[
                "The map appears after map selection",
                "Match filters narrow the current scope",
                "Event Investigation focuses exact raw events",
            ],
        )
        st.success(f"Loaded {loaded_count} files from {selected_date} (failed: {failed_count}).")
        return

    map_df = df[df["map_id"] == selected_map].copy()
    if map_df.empty:
        st.warning("No rows found for the selected map.")
        return

    # Sidebar: Match filter
    st.sidebar.markdown("### Match Filter")

    available_matches = sorted(map_df["match_id"].dropna().astype(str).unique().tolist())

    match_search = st.sidebar.text_input(
        "Search Match IDs",
        value="",
        help="Type a full or partial match ID. If nothing is explicitly selected, the search itself filters the scope.",
    ).strip().lower()

    visible_matches = (
        [m for m in available_matches if match_search in m.lower()]
        if match_search
        else available_matches
    )

    search_has_no_results = bool(match_search) and len(visible_matches) == 0

    select_all_matches = st.sidebar.checkbox(
        "Select All Visible",
        value=False,
        disabled=(len(visible_matches) == 0),
    )

    selected_matches = st.sidebar.multiselect(
        "Match IDs",
        options=visible_matches,
        default=[],
        disabled=(len(visible_matches) == 0),
    )

    if select_all_matches:
        selected_matches = visible_matches
        st.sidebar.caption(f"Selected {len(visible_matches)} visible matches")

    if selected_matches:
        active_match_ids = selected_matches
        match_filter_mode = "explicit_selection"
    elif match_search:
        active_match_ids = visible_matches
        match_filter_mode = "search_filter"
    else:
        active_match_ids = []
        match_filter_mode = "none"

    selected_matches_active = bool(selected_matches) or bool(match_search)

    match_df = (
        map_df[map_df["match_id"].astype(str).isin(active_match_ids)].copy()
        if selected_matches_active
        else pd.DataFrame(columns=map_df.columns)
    )

    # Sidebar: Event Investigation
    st.sidebar.markdown("### Event Investigation")
    event_focus = st.sidebar.selectbox("Focused Event", FOCUS_EVENT_OPTIONS, index=0)
    show_only_focused_events = st.sidebar.checkbox(
        "Strict Focus Mode",
        value=True,
        help="When on, exact focused events are shown without extra movement context unless you are already filtering to matches.",
    )

    # Sidebar: Display options
    with st.sidebar.expander("Display Options", expanded=False):
        include_human_journeys = st.checkbox("Include Human Journeys", value=True)
        include_bot_journeys = st.checkbox("Include Bot Journeys", value=True)
        show_lines = st.checkbox("Show Movement Paths", value=True)
        show_event_markers = st.checkbox("Show Event Markers", value=True)
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        color_by_match = st.checkbox("Color by Match", value=False)
        hide_out_of_bounds = st.checkbox("Hide Out-of-Bounds", value=False)
        max_players_per_match = st.slider(
            "Max players per match",
            min_value=5,
            max_value=80,
            value=25,
            step=5,
        )
        hotspot_cell_size = st.slider(
            "Hotspot Grid Size",
            min_value=16,
            max_value=128,
            value=48,
            step=8,
        )

    # Sidebar: Playback
    with st.sidebar.expander("Playback", expanded=False):
        if selected_matches_active and not search_has_no_results:
            enable_timeline = st.checkbox("Enable Timeline", value=False)
            progress_pct = 100

            if enable_timeline:
                progress_pct = st.slider("Playback %", 0, 100, 100, 5)
                match_df, start_time, end_time, cutoff_time = apply_timeline_filter(match_df, progress_pct)
            else:
                start_time = match_df["event_time"].min() if not match_df.empty else None
                end_time = match_df["event_time"].max() if not match_df.empty else None
                cutoff_time = None
        else:
            enable_timeline = False
            progress_pct = 100
            start_time = None
            end_time = None
            cutoff_time = None
            if selected_matches_active and search_has_no_results:
                st.caption("Playback is unavailable because no matches match the current search.")
            else:
                st.caption("Playback becomes available after a match filter is active.")

    # Sidebar: Map calibration
    cfg = MAP_CONFIG[selected_map]
    with st.sidebar.expander("Map Calibration", expanded=False):
        origin_x = st.number_input("origin_x", value=float(cfg["origin_x"]), step=10.0, format="%.2f")
        origin_z = st.number_input("origin_z", value=float(cfg["origin_z"]), step=10.0, format="%.2f")
        scale = st.number_input("scale", value=float(cfg["scale"]), step=10.0, min_value=1.0, format="%.2f")

    # Coordinate mapping
    mapped_day_map_df = map_coordinates(map_df, origin_x=origin_x, origin_z=origin_z, scale=scale)
    if mapped_day_map_df.empty:
        st.warning("No plottable rows available for the selected date + map.")
        return

    day_map_pre_filter_stats = pixel_stats(mapped_day_map_df)

    if hide_out_of_bounds:
        mapped_day_map_df = mapped_day_map_df[mapped_day_map_df["in_bounds"]].copy()
        if mapped_day_map_df.empty:
            st.warning("All day-level points are outside the minimap bounds.")
            return

    if selected_matches_active:
        mapped_match_df = map_coordinates(match_df, origin_x=origin_x, origin_z=origin_z, scale=scale)
        selected_match_pre_filter_stats = pixel_stats(mapped_match_df)

        if hide_out_of_bounds and not mapped_match_df.empty:
            mapped_match_df = mapped_match_df[mapped_match_df["in_bounds"]].copy()
    else:
        mapped_match_df = pd.DataFrame(columns=mapped_day_map_df.columns)
        selected_match_pre_filter_stats = {}

    # Build active and analysis scopes
    base_scope_df = mapped_match_df.copy() if selected_matches_active else mapped_day_map_df.copy()

    focused_event_label = None
    active_scope_df = base_scope_df.copy()

    if event_focus != "All Events":
        focused_event_label = event_focus
        focused_df = apply_event_focus_filter(base_scope_df, event_focus)

        if show_only_focused_events:
            if selected_matches_active and show_lines:
                movement_part = base_scope_df[base_scope_df["event_group"] == "movement"].copy()
                active_scope_df = pd.concat([movement_part, focused_df], ignore_index=True).drop_duplicates()
            else:
                active_scope_df = focused_df.copy()
        else:
            active_scope_df = base_scope_df.copy()

    active_scope_df = filter_by_player_type(
        active_scope_df,
        include_human_journeys=include_human_journeys,
        include_bot_journeys=include_bot_journeys,
    )

    if event_focus != "All Events":
        analysis_df = apply_event_focus_filter(base_scope_df, event_focus)
        analysis_df = filter_by_player_type(
            analysis_df,
            include_human_journeys=include_human_journeys,
            include_bot_journeys=include_bot_journeys,
        )
    else:
        analysis_df = active_scope_df.copy()

    try:
        base_map = load_minimap(selected_map)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    # Overview
    total_rows = len(active_scope_df)
    movement_rows = int((active_scope_df["event_group"] == "movement").sum()) if not active_scope_df.empty else 0
    kill_rows = int((active_scope_df["event_group"] == "kill").sum()) if not active_scope_df.empty else 0
    death_rows = int((active_scope_df["event_group"] == "death").sum()) if not active_scope_df.empty else 0
    loot_rows = int((active_scope_df["event_group"] == "loot").sum()) if not active_scope_df.empty else 0

    human_journey_count = (
        active_scope_df.loc[active_scope_df["player_type"] == "Human", "user_id_str"].nunique()
        if not active_scope_df.empty else 0
    )
    bot_journey_count = (
        active_scope_df.loc[active_scope_df["player_type"] == "Bot", "user_id_str"].nunique()
        if not active_scope_df.empty else 0
    )

    focus_count = len(analysis_df)

    st.subheader("Overview")

    metric_cols_top = st.columns(3)
    with metric_cols_top[0]:
        render_metric_card("Rows in View", f"{total_rows:,}")
    with metric_cols_top[1]:
        render_metric_card("Matches Filtered", f"{len(active_match_ids):,}")
    with metric_cols_top[2]:
        render_metric_card("Focused Rows", f"{focus_count:,}")

    metric_cols_bottom = st.columns(4)
    with metric_cols_bottom[0]:
        render_metric_card("Human / Bot Journeys", f"{human_journey_count:,} / {bot_journey_count:,}")
    with metric_cols_bottom[1]:
        render_metric_card("Kill-side Events", f"{kill_rows:,}")
    with metric_cols_bottom[2]:
        render_metric_card("Death-side Events", f"{death_rows:,}")
    with metric_cols_bottom[3]:
        render_metric_card("Loot", f"{loot_rows:,}")

    if selected_matches_active and enable_timeline and not search_has_no_results:
        st.info(f"Playback at {progress_pct}%. Showing data up to {format_ts(cutoff_time)}.")
        st.caption(f"Match time window: {format_ts(start_time)} → {format_ts(end_time)}")
    elif selected_matches_active and not search_has_no_results:
        st.caption(f"Match time window: {format_ts(start_time)} → {format_ts(end_time)}")

    st.caption(f"Current scope: {get_scope_label(selected_matches_active, event_focus, match_filter_mode)}")

    if search_has_no_results:
        st.warning("No match IDs match the current search for this date and map. Showing an empty result set.")

    if event_focus != "All Events" and analysis_df.empty and not search_has_no_results:
        st.warning(
            f"No rows match '{event_focus}' in the current date, map, match scope, and journey filters."
        )

    # Map Analysis
    st.subheader("Map Analysis")

    fig = build_plotly_figure(
        base_map=base_map,
        df=active_scope_df,
        max_players_per_match=max_players_per_match,
        show_lines=show_lines,
        show_event_markers=show_event_markers,
        show_heatmap=show_heatmap,
        color_by_match=color_by_match,
        focused_event_label=focused_event_label,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
                "autoScale2d",
            ],
        },
    )

    legend_left, legend_right = st.columns(2)

    with legend_left:
        if color_by_match:
            st.markdown(
                """
                - **Color**: match-based coloring for visible matches  
                - **Solid path + circle endpoint**: human journey  
                - **Dotted path + triangle endpoint**: bot journey
                """
            )
        else:
            lines = []
            if include_human_journeys:
                lines.append("- **Green solid path + circle endpoint**: human journey movement")
            if include_bot_journeys:
                lines.append("- **Blue dotted path + triangle endpoint**: bot journey movement")
            if not lines:
                lines.append("- Movement paths are currently hidden by journey filters")
            st.markdown("\n".join(lines))

    with legend_right:
        marker_lines = []
        if show_event_markers:
            marker_lines.extend(
                [
                    "- **Red X**: `kill` → human killed human",
                    "- **Orange X**: `botkill` → human killed a bot",
                    "- **Yellow diamond**: `killed` → human killed by human",
                    "- **Blue diamond**: `botkilled` → human killed by bot",
                    "- **Silver diamond**: `killedbystorm` → killed by storm",
                    "- **Purple square**: `loot` → loot pickup",
                ]
            )
        else:
            marker_lines.append("- Event markers are currently off")

        if show_heatmap:
            marker_lines.append("- Heatmap shows density of currently visible points")
        else:
            marker_lines.append("- Heatmap is currently off")

        if focused_event_label:
            marker_lines.append("- White/Cyan ring highlights the currently focused event")

        st.markdown("\n".join(marker_lines))

    st.caption(
        "This tool analyzes journey files. 'Human' and 'Bot' filters refer to the journey owner for each file, not necessarily the actor/victim role of every event."
    )

    # Diagnostics
    stats = pixel_stats(active_scope_df)
    d1, d2 = st.columns(2)

    with d1:
        st.subheader("Event Mix")
        st.write(
            {
                "movement": movement_rows,
                "kill_side": kill_rows,
                "death_side": death_rows,
                "loot": loot_rows,
                "other": int((active_scope_df["event_group"] == "other").sum()) if not active_scope_df.empty else 0,
            }
        )

    with d2:
        st.subheader("Pixel Mapping Diagnostics")
        st.write(
            {
                "pixel_x_min": round(stats.get("pixel_x_min", 0.0), 2),
                "pixel_x_max": round(stats.get("pixel_x_max", 0.0), 2),
                "pixel_y_min": round(stats.get("pixel_y_min", 0.0), 2),
                "pixel_y_max": round(stats.get("pixel_y_max", 0.0), 2),
                "in_bounds_current_view": stats.get("in_bounds", 0),
                "out_of_bounds_current_match_filter_before_hide": selected_match_pre_filter_stats.get("out_of_bounds", 0),
                "out_of_bounds_day_map_before_hide": day_map_pre_filter_stats.get("out_of_bounds", 0),
                "total_rows_current_view": stats.get("total_rows", 0),
            }
        )

    # Investigation Tables
    st.subheader("Investigation Tables")

    st.markdown("#### Event Hotspot Summary")
    hotspot_source_df = analysis_df.copy()

    if event_focus == "All Events":
        hotspot_source_df = hotspot_source_df[
            hotspot_source_df["event_group"].isin({"kill", "death", "loot"})
        ].copy()

    hotspot_summary_df = build_hotspot_summary(
        df=hotspot_source_df,
        cell_size=hotspot_cell_size,
    )

    if hotspot_summary_df.empty:
        if search_has_no_results:
            st.info("No hotspot data because the current match search returned no matches.")
        else:
            st.info("No hotspot data available for the current scope.")
    else:
        st.dataframe(
            hotspot_summary_df.head(15),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Focused Event Match IDs")
    if event_focus == "All Events":
        st.info("Select a focused event to see which match IDs are involved.")
    else:
        focused_match_table = build_focused_match_table(analysis_df)

        if focused_match_table.empty:
            if search_has_no_results:
                st.info("No focused-event matches because the current match search returned no matches.")
            else:
                st.info("No matching events found in the current scope.")
        else:
            st.dataframe(
                focused_match_table,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Use these match IDs in Match Filter to inspect movement paths and playback for the relevant matches."
            )

    st.markdown("#### Match Summary")
    if selected_matches_active and not mapped_match_df.empty:
        summary_df = build_match_summary(mapped_match_df)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    elif search_has_no_results:
        st.info("No match summary because the current match search returned no matches.")
    else:
        st.info("No match filter is active. Type in Search Match IDs or choose Match IDs to narrow the view.")

    # Raw Event Evidence
    with st.expander("Raw Event Evidence"):
        event_table = build_event_table(analysis_df)

        if event_table.empty:
            if search_has_no_results:
                st.write("No rows to show because the current match search returned no matches.")
            else:
                st.write("No events available in the current analysis scope.")
        else:
            st.dataframe(event_table, use_container_width=True, hide_index=True)

    st.success(f"Loaded {loaded_count} files from {selected_date} (failed: {failed_count}).")


if __name__ == "__main__":
    main()