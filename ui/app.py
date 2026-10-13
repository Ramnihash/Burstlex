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


@st.cache_data(show_spinner=False)
def process(data, min_frequency):
    return run_pipeline(data, min_frequency)


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
    with st.spinner("Running lexer, segmentation, clustering and optimizer..."):
        st.session_state["result"] = process(data, int(min_frequency))
        st.session_state["tweets"] = len(data)

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

overview_tab, events_tab, passes_tab = st.tabs(
    ["Overview", "Events", "Compiler passes"]
)

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
