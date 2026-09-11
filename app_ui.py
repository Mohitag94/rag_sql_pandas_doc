import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Pandas AI Assistant",
    page_icon="🐼",
    layout="centered",
)

# Constant API configuration
API_URL = "https://pandas-rag-api-mka-576093593374.us-central1.run.app/api/v1/ask"

# Sidebar Branding
with st.sidebar:
    st.markdown("## 🐼 Pandas RAG Engine")
    st.markdown(
        "Ask technical questions regarding data handling, DataFrame operations, "
        "merges, groupings, or cleaning protocols."
    )
    st.caption("Backend Infrastructure: Llama-3.1-8B-Instruct + FAISS")

    st.markdown("### 📚 Knowledge Base Scope")
    with st.expander("🔍 View Covered Topics (11 Guides)", expanded=False):
        st.markdown(
            "The index is explicitly trained on these 11 core Pandas documentation chapters:\n\n"
            "* ⏱️ **10min** (Quick Start User Guide)\n"
            "* 📋 **basics** (Essential functionality)\n"
            "* 🗄️ **groupby** (Split-Apply-Combine operations)\n"
            "* 🎯 **indexing** (Indexing and selecting data)\n"
            "* 💾 **io** (Input/Output tools & storage file formats)\n"
            "* 🔗 **merging** (Merge, join, concatenate, and compare)\n"
            "* 🧼 **missing_data** (Working with missing / NaN records)\n"
            "* 🔄 **reshaping** (Reshaping and pivot tables)\n"
            "* 🔤 **text** (Working with text / string columns)\n"
            "* 📈 **timeseries** (Time series & date functionalities)\n"
            "* 📊 **visualization** (Plotting and data graphics)"
        )
    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main interface layout
st.title("🐼 Pandas Docs Assistant")
st.markdown("---")

# Initialize session history tracking state
if "history" not in st.session_state:
    st.session_state.history = []

# Question interaction block
question = st.text_input(
    "Ask a question about pandas:",
    placeholder="e.g., How do I merge two DataFrames on index matches?",
)

st.warning(
    "**Note on Token Constraints:** To ensure optimal speeds, answers are bound by "
    "server token thresholds. If your query demands an exceptionally long explanation "
    "or complex code generation, the output may occasionally cut off mid-sentence.",
    icon="⚠️",
)

if (
    st.button("⚡ Query Assistant", type="primary", use_container_width=True)
    and question
):
    # Use a clean spinner context manager to improve user feedback UX
    with st.spinner("Processing semantics and fetching documentation nodes..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=30)
            if response.status_code == 200:
                result = response.json()

                # Insert incoming records at top of the stack
                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "answer": result.get("answer", "No context returned."),
                        "confidence_score": result.get("confidence_score"),
                    },
                )
                # Keep rolling stack historical limit bound to 5 entries max
                st.session_state.history = st.session_state.history[:5]
            else:
                st.error(f"Engine connection failure (HTTP {response.status_code}).")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to communicate with backplane network: {e!s}")

# Display historical card stack nicely if metrics populate
if st.session_state.history:
    st.markdown("### 💬 Recent Queries")

    for entry in st.session_state.history:
        # Use native chat message containers for high quality UI card formatting
        with st.chat_message("user"):
            st.markdown(f"**{entry['question']}**")

        with st.chat_message("assistant", avatar="🐼"):
            st.markdown(entry["answer"])

            # Use visual columns to align meta statistics cleanly under text blocks
            if entry["confidence_score"] is not None:
                col1, _ = st.columns([1, 3])
                with col1:
                    st.metric(
                        label="Confidence Alignment",
                        value=f"{entry['confidence_score']:.3f}",
                    )
        st.markdown("---")
