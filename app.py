import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from isaga_core import (
    IndusSignDatabase, 
    IndusInscription, 
    CorpusAnalyzer, 
    PredictiveRepairEngine,
    IndusNetworkVisualizer,
    prepare_streamlit_app
)

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Harappa Valley | ISAGA 2.0",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .valid-box { padding: 10px; background-color: #d4edda; border-radius: 5px; color: #155724; }
    .invalid-box { padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24; }
    .predict-box { padding: 15px; background-color: #e2e3e5; border-radius: 5px; border-left: 5px solid #383d41; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'scientific_mode' not in st.session_state:
    st.session_state.scientific_mode = False
if 'analyzer' not in st.session_state:
    # Initialize the AI and train it on the "Simulated Corpus" immediately
    analyzer = CorpusAnalyzer()
    # Training data (Simulated from Mahadevan)
    training_data = [
        [59, 99, 342], [211, 99, 342], [123, 456, 342], 
        [59, 789, 342], [211, 789, 342], [65, 99, 343]
    ]
    for seq in training_data:
        analyzer.add_inscription(IndusInscription(seq))
    st.session_state.analyzer = analyzer

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧱 Harappa Valley")
    st.caption("Indus Script Administrative Protocol Analyzer")
    
    mode = st.radio("Select Module:", 
        ["🛠️ Protocol Builder", "🧩 Broken Seal Predictor", "🕸️ Network Visualizer"])
    
    st.divider()
    
    # The Scientific Toggle
    st.session_state.scientific_mode = st.toggle("Scientific Mode", value=False)
    if st.session_state.scientific_mode:
        st.info("🔬 Showing Z-Scores, Bayesian probabilities, and Entropy values.")
    else:
        st.caption("Showing simplified view.")

    st.divider()
    st.markdown("### 📚 Reference")
    st.markdown("**342 (Jar):** Terminal Seal\n\n**343 (Marked Jar):** Input Start\n\n**59 (Fish):** Commodity\n\n**99 (Arrow):** Operator")

# --- LOAD DATA ---
data = prepare_streamlit_app()
db = IndusSignDatabase()

# ==========================================
# PAGE 1: PROTOCOL BUILDER (The Game)
# ==========================================
if mode == "🛠️ Protocol Builder":
    st.header("🛠️ Administrative Protocol Builder")
    st.markdown("Build a valid transaction receipt using the standard Harappan signs.")

    # 1. The Sign Picker (Visual Grid)
    st.subheader("Select Signs")
    
    # Display signs in columns
    cols = st.columns(6)
    for idx, sign in enumerate(data['sign_catalog']):
        col = cols[idx % 6]
        with col:
            if st.button(f"{sign['name']}\n({sign['id']})", key=sign['id']):
                st.session_state.sequence.append(sign['id'])

    # 2. The Current Tablet
    st.divider()
    st.subheader("Current Sequence")
    
    if st.session_state.sequence:
        # Display the sequence visually
        seq_cols = st.columns(len(st.session_state.sequence) + 1)
        for i, s_id in enumerate(st.session_state.sequence):
            with seq_cols[i]:
                st.button(db.get_name(s_id), key=f"seq_{i}", disabled=True)
        
        # Clear Button
        if st.button("Clear Tablet"):
            st.session_state.sequence = []
            st.rerun()

        # 3. The Validation Engine
        st.divider()
        ins = IndusInscription(st.session_state.sequence)
        result = ins.validate_syntax()
        
        if result['valid']:
            st.markdown(f"""<div class="valid-box">✅ <b>VALID PROTOCOL</b><br>{result['message']}</div>""", unsafe_allow_html=True)
            if st.session_state.scientific_mode:
                st.write(f"Sequence Length: {len(st.session_state.sequence)}")
                st.write("Transition Probability: 0.98 (Simulated)")
        else:
            st.markdown(f"""<div class="invalid-box">❌ <b>PROTOCOL VIOLATION</b><br>{result['error']}</div>""", unsafe_allow_html=True)

    else:
        st.info("Tablet is empty. Select signs above.")

# ==========================================
# PAGE 2: BROKEN SEAL PREDICTOR (The AI)
# ==========================================
elif mode == "🧩 Broken Seal Predictor":
    st.header("🧩 Broken Seal Reconstruction")
    st.markdown("Use the Bayesian Engine to predict missing signs in a damaged inscription.")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        pre_sign = st.selectbox("Preceding Sign (Left)", ["START"] + [s['name'] for s in data['sign_catalog']])
    with col2:
        st.markdown("<h2 style='text-align: center; color: red;'>?</h2>", unsafe_allow_html=True)
        st.caption("Missing Fragment")
    with col3:
        post_sign = st.selectbox("Following Sign (Right)", ["END"] + [s['name'] for s in data['sign_catalog']])

    if st.button("Run Prediction Engine"):
        # Map names back to IDs
        pre_id = next((s['id'] for s in data['sign_catalog'] if s['name'] == pre_sign), None)
        post_id = next((s['id'] for s in data['sign_catalog'] if s['name'] == post_sign), None)
        
        # Construct the broken sequence for the engine
        if pre_sign == "START":
            seq = [None, post_id]
            gap_idx = 0
        else:
            seq = [pre_id, None, post_id] if post_sign != "END" else [pre_id, None]
            gap_idx = 1
            
        # Run AI
        predictor = PredictiveRepairEngine(st.session_state.analyzer)
        predictions = predictor.predict_missing_sign(seq, gap_idx)
        
        st.subheader("AI Confidence Report")
        
        if not predictions:
            st.warning("No statistically viable bridge found between these signs.")
        
        for rank, (s_id, conf, logic) in enumerate(predictions, 1):
            name = db.get_name(s_id)
            
            # Visual Bar for Confidence
            st.markdown(f"**{rank}. {name}** (ID: {s_id})")
            st.progress(conf)
            
            if st.session_state.scientific_mode:
                st.markdown(f"""<div class="predict-box">
                <b>Bayesian Logic:</b><br>
                {logic}<br>
                Grammar Check: {db.get_role(s_id)}
                </div>""", unsafe_allow_html=True)
            else:
                st.caption(f"Confidence: {int(conf*100)}%")

# ==========================================
# PAGE 3: NETWORK VISUALIZER (The Graph)
# ==========================================
elif mode == "🕸️ Network Visualizer":
    st.header("🕸️ Administrative Flow Network")
    st.markdown("Visualizing the 'River System' of the Indus Script.")
    
    visualizer = IndusNetworkVisualizer(st.session_state.analyzer)
    
    if st.button("Generate Live Graph"):
        with st.spinner("Calculating centrality metrics..."):
            fig, ax = plt.subplots(figsize=(10, 8))
            G = visualizer.graph
            pos = nx.spring_layout(G, k=2)
            
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=1500, font_size=10, font_weight='bold', 
                    edge_color='gray', width=1.5, ax=ax)
            
            st.pyplot(fig)
            
            if st.session_state.scientific_mode:
                st.subheader("Network Metrics")
                metrics = visualizer.analyze_network_properties()
                st.json(metrics)

# --- FOOTER ---
st.divider()
st.caption("© 2025 IndusLogic | Built on ISAGA 2.0 | [View Source](https://github.com/IndusLogic)")
