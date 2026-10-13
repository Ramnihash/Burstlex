import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline import run_pipeline

BUNDLED = {
    "Election 2024 (64,044 tweets)": ("data/election_tweets.csv", 20),
    "Sample (24 tweets)": ("data/tweets.csv", 2),
}

st.set_page_config(
    page_title="Burstlex",
    page_icon="📡",
    layout="wide"
)


TIMELINE_CSS = """
<style>
.bx{--bx-accent:#3b82f6;--bx-line:rgba(128,128,128,.28);--bx-surface:rgba(128,128,128,.07);
font-size:16px;line-height:1.5;max-width:900px}
.bx-step{display:grid;grid-template-columns:44px 1fr;gap:0 16px}
.bx-rail{display:flex;flex-direction:column;align-items:center}
.bx-num{width:36px;height:36px;border-radius:50%;background:var(--bx-accent);color:#fff;
display:flex;align-items:center;justify-content:center;font-weight:600;font-size:15px;flex:none}
.bx-num svg{width:18px;height:18px}
.bx-stem{width:2px;flex:1;background:var(--bx-line);margin:4px 0}
.bx-step:last-child .bx-stem{display:none}
.bx-card{border:1px solid var(--bx-line);border-radius:12px;padding:14px 18px;
margin-bottom:16px;background:var(--bx-surface);min-width:0}
.bx-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.bx-name{font-weight:600;font-size:17px}
.bx-time{font-size:14px;opacity:.75;font-variant-numeric:tabular-nums}
.bx-metric{display:flex;align-items:baseline;gap:8px;margin:6px 0 2px}
.bx-value{font-size:28px;font-weight:700;color:var(--bx-accent);font-variant-numeric:tabular-nums}
.bx-label{font-size:15px;opacity:.8}
.bx-sum{font-size:14px;opacity:.75;margin-bottom:8px}
.bx-bar{height:6px;border-radius:3px;background:var(--bx-line);overflow:hidden}
.bx-bar i{display:block;height:100%;background:var(--bx-accent);border-radius:3px}
.bx-sub{font-size:13px;opacity:.7;margin:12px 0 6px;text-transform:uppercase;letter-spacing:.04em}
.bx-mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13.5px;
padding:8px 12px;border-radius:8px;border:1px solid var(--bx-line);margin-bottom:6px;
white-space:pre-wrap;overflow-wrap:anywhere}
.bx-tag{opacity:.65;font-size:12px;display:block;margin-bottom:2px}
.bx-chips{display:flex;flex-wrap:wrap;gap:6px}
.bx-chip{border:1px solid var(--bx-line);border-radius:999px;padding:2px 12px;font-size:14px;
background:var(--bx-surface)}
</style>
"""

def chips(items):
    return '<div class="bx-chips">' + "".join(
        f'<span class="bx-chip">{html.escape(str(item))}</span>' for item in items
    ) + "</div>"


def stage_extra(stage):
    details = stage["details"] or {}
    name = stage["name"]

    if name == "Lexer" and details.get("example"):
        example = details["example"]
        return (
            '<div class="bx-sub">Before and after on a real tweet</div>'
            f'<div class="bx-mono"><span class="bx-tag">Raw</span>{html.escape(example["before"])}</div>'
            f'<div class="bx-mono"><span class="bx-tag">Cleaned</span>{html.escape(example["after"])}</div>'
        )

    if name == "Segmentation" and details.get("example"):
        return '<div class="bx-sub">Segments from the first tweet</div>' + chips(details["example"])

    if name == "Adaptive symbol table" and details.get("learned"):
        return '<div class="bx-sub">Emerging phrases learned</div>' + chips(details["learned"][:20])

    return ""


def pipeline_timeline(stages):
    longest = max((stage["seconds"] for stage in stages), default=0) or 1
    parts = [TIMELINE_CSS, '<div class="bx">']

    for number, stage in enumerate(stages, 1):
        share = max(stage["seconds"] / longest * 100, 2)
        parts.append(
            f'<div class="bx-step"><div class="bx-rail">'
            f'<div class="bx-num" title="Stage {number} complete">{number}</div>'
            f'<div class="bx-stem"></div></div>'
            f'<div class="bx-card">'
            f'<div class="bx-head"><span class="bx-name">{html.escape(stage["name"])}</span>'
            f'<span class="bx-time">{stage["seconds"]:.1f}s</span></div>'
            f'<div class="bx-metric"><span class="bx-value">{html.escape(str(stage["headline"]))}</span>'
            f'<span class="bx-label">{html.escape(stage["label"])}</span></div>'
            f'<div class="bx-sum">{html.escape(stage["summary"])}</div>'
            f'<div class="bx-bar" role="img" aria-label="{stage["seconds"]:.1f} seconds">'
            f'<i style="width:{share:.0f}%"></i></div>'
            f'{stage_extra(stage)}</div></div>'
        )

    parts.append("</div>")
    return "".join(parts)


STAGE_COUNT = 10


def run_with_tracker(data, min_frequency):
    done = []

    with st.status("Running pipeline...", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")

        def show(record):
            done.append(record)
            st.write(
                f"**{len(done)}. {record['name']}** · "
                f"{record['headline']} {record['label']} · "
                f"{record['seconds']:.1f}s"
            )
            bar.progress(
                min(len(done) / STAGE_COUNT, 1.0),
                text=f"{len(done)} of {STAGE_COUNT} stages complete"
            )

        outcome = run_pipeline(data, min_frequency, on_stage=show)
        status.update(
            label="Pipeline complete", state="complete", expanded=False
        )

    return outcome


def events_table(events, novelty):
    return pd.DataFrame([
        {
            "Rank": rank,
            "Event": event["summary"],
            "Tweets": event["frequency"],
            "Burst score": round(event["score"], 2),
            "Status": novelty.get(event["summary"], {}).get("status", "n/a"),
            "Segments": len(event["segments"]),
        }
        for rank, event in enumerate(events, 1)
    ])


with st.sidebar:
    st.title("Burstlex")
    st.caption("Event detection from tweets")

    source = st.radio("Data source", ["Bundled dataset", "Upload CSV"])

    data = None
    default_frequency = 20

    if source == "Bundled dataset":
        choice = st.selectbox("Dataset", list(BUNDLED))
        path, default_frequency = BUNDLED[choice]
        data = pd.read_csv(ROOT / path)
    else:
        uploaded = st.file_uploader("Tweets CSV", type=["csv"])
        st.caption("Columns: tweet_id, text, timestamp")

        if uploaded is not None:
            data = pd.read_csv(uploaded)

    min_frequency = st.slider(
        "Minimum segment frequency",
        2, 100, default_frequency,
        help="Higher values keep only segments that appear in more tweets."
    )

    run = st.button("Detect events", type="primary")

st.title("Burstlex: Event Detection from Tweets")
st.write(
    "Segmentation-based event detection with an error-recovering lexer, "
    "an adaptive symbol table and an event optimizer."
)

if data is None:
    st.info("Choose a dataset in the sidebar, then press **Detect events**.")
    st.stop()

missing = {"tweet_id", "text", "timestamp"} - set(data.columns)

if missing:
    st.error("Missing columns: " + ", ".join(sorted(missing)))
    st.stop()

if run:
    st.session_state["result"] = run_with_tracker(data, int(min_frequency))

result = st.session_state.get("result")

if result is None:
    st.info(
        f"{len(data):,} tweets loaded. Press **Detect events** in the sidebar."
    )
    st.stop()

events = result["events"]
evaluation = result["evaluation"]
novelty = result["novelty"] or {}
optimizer_info = result["optimizer_info"]
symbol_table = result["symbol_table"]
lexical_errors = result["lexical_errors"]

pipeline_tab, overview_tab, events_tab, passes_tab = st.tabs(
    ["Pipeline", "Overview", "Events", "Compiler passes"]
)

with pipeline_tab:
    stages = result["stages"]
    total = sum(stage["seconds"] for stage in stages)

    st.caption(
        f"{len(stages)} stages, {total:.1f}s in total. Each stage feeds the next."
    )
    st.html(pipeline_timeline(stages))

with overview_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tweets", f"{evaluation['total_tweets']:,}")
    c2.metric("Events", evaluation["total_events"])
    c3.metric("Event coverage", f"{evaluation['event_coverage'] * 100:.1f}%")
    c4.metric("Avg burst score", round(evaluation["average_burst_score"], 2))

    st.subheader("Top events by tweets")

    if events:
        top = events_table(events, novelty).head(15)
        st.bar_chart(
            top, x="Event", y="Tweets", horizontal=True, sort="-Tweets"
        )
    else:
        st.warning("No events detected. Try a lower minimum frequency.")

with events_tab:
    if not events:
        st.warning("No events detected.")
    else:
        table = events_table(events, novelty)
        search = st.text_input("Search events", placeholder="e.g. biden")
        statuses = sorted(table["Status"].unique())
        chosen = st.multiselect("Status", statuses, default=statuses)

        view = table[table["Status"].isin(chosen)]

        if search:
            view = view[
                view["Event"].str.contains(search, case=False, regex=False)
            ]

        st.caption(f"{len(view)} of {len(table)} events")
        st.dataframe(view, hide_index=True)

        st.subheader("Event details")
        pick = st.selectbox("Event", view["Event"].tolist() or [None])

        if pick:
            event = next(e for e in events if e["summary"] == pick)
            d1, d2, d3 = st.columns(3)
            d1.metric("Tweets", event["frequency"])
            d2.metric("Burst score", round(event["score"], 2))
            d3.metric(
                "Status", novelty.get(pick, {}).get("status", "n/a")
            )
            st.write("**Related segments:**", ", ".join(event["segments"]))

with passes_tab:
    left, right = st.columns(2)

    with left:
        st.subheader("1. Error-recovering lexer")
        st.metric("Lexical errors recovered", f"{sum(lexical_errors.values()):,}")

        if lexical_errors:
            errors = pd.DataFrame(
                lexical_errors.most_common(), columns=["Error", "Count"]
            )
            st.bar_chart(
                errors, x="Error", y="Count", horizontal=True, sort="-Count"
            )

    with right:
        st.subheader("2. Adaptive symbol table")
        st.metric("Emerging phrases learned", len(symbol_table.learned))

        if symbol_table.learned:
            st.dataframe(
                pd.DataFrame(
                    {"Learned phrase": sorted(symbol_table.learned)}
                ),
                hide_index=True,
                height=260,
            )

    st.subheader("3. Event optimizer")

    if optimizer_info:
        o1, o2, o3 = st.columns(3)
        o1.metric("Before", optimizer_info["before"])
        o2.metric(
            "After merging duplicates",
            optimizer_info["after_merge"],
            optimizer_info["after_merge"] - optimizer_info["before"],
            delta_color="off"
        )
        o3.metric(
            "After removing noise",
            optimizer_info["after_noise"],
            optimizer_info["after_noise"] - optimizer_info["after_merge"],
            delta_color="off"
        )

        with st.expander("Events removed as noise"):
            st.write(", ".join(optimizer_info["dropped"]) or "None")
    else:
        st.caption("Optimizer is skipped for datasets without timestamps.")
