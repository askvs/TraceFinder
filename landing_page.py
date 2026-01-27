import os
import streamlit as st
import time
import random
from PIL import Image
import numpy as np
import pandas as pd

# Model inference module
from model_inference import (
    predict_hybrid_cnn,
    predict_cnn,
    predict_baseline,
    generate_gradcam,
    check_models_available
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trace Finder - Forensic Scanner Identification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Modern UI & CSS Styling (Includes Sidebar & Box Styles)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* GLOBAL RESET */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0; 
    }
    
    /* HEADINGS */
    h1 { font-size: 3.5rem !important; font-weight: 900; letter-spacing: -1px; }
    h2 { font-size: 2.2rem !important; font-weight: 700; color: #00d4ff !important; margin-top: 2rem; }
    h3 { font-size: 1.4rem !important; font-weight: 600; color: #ffffff; }
    p  { font-size: 1rem; line-height: 1.6; color: #b0b0b0; }

    /* HERO SECTION */
    .hero-container {
        background: radial-gradient(circle at 50% 0%, rgba(0, 212, 255, 0.15) 0%, rgba(14, 17, 23, 1) 80%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeIn 1.2s ease-out;
    }
    .hero-title {
        background: linear-gradient(90deg, #ffffff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* FEATURE CARDS */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        background: rgba(255, 255, 255, 0.08);
    }

    /* METRICS */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d4ff;
    }
    
    /* =========================================
       SIDEBAR CUSTOM STYLING
       ========================================= */
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Remove default bullets from markdown lists in sidebar */
    [data-testid="stSidebar"] ul {
        list-style-type: none;
        padding-left: 0;
    }

    /* Style the links to look like menu buttons */
    [data-testid="stSidebar"] a {
        text-decoration: none;
        color: #b0b0b0 !important;
        font-weight: 500;
        display: block;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }

    /* Hover effect for links */
    [data-testid="stSidebar"] a:hover {
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff !important;
        border-left: 3px solid #00d4ff;
        padding-left: 20px; /* Slight movement effect */
    }
    
    /* Separator lines in sidebar */
    [data-testid="stSidebar"] hr {
        margin: 2rem 0;
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* ANIMATION */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar 
# -----------------------------------------------------------------------------
with st.sidebar:
    # --- LOGO & BRANDING ---
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="
            display: inline-flex; 
            align-items: center; 
            justify-content: center; 
            width: 60px; 
            height: 60px; 
            background: linear-gradient(135deg, #00d4ff, #004e66); 
            border-radius: 15px; 
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        ">
            <span style="font-size: 30px;">🔍</span>
        </div>
        <h2 style="color: #fff !important; margin: 0; font-size: 1.5rem !important;">Trace Finder</h2>
        <p style="color: #666; font-size: 0.8rem; letter-spacing: 1px;">FORENSIC SCANNER ID</p>
    </div>
    """, unsafe_allow_html=True)

    # --- NAVIGATION MENU ---
    st.markdown("""
    <p style="font-size: 0.75rem; color: #555; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">MAIN MENU</p>
    
    <a href="#trace-finder">🏠 &nbsp; Dashboard Home</a>
    <a href="#project-overview">📋 &nbsp; Project Overview</a>
    <a href="#system-architecture">⚙️ &nbsp; System Architecture</a>
    <a href="#forensic-analysis-lab">🧪 &nbsp; Live Analysis Lab</a>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SYSTEM STATUS PANEL ---
    st.markdown("""
    <p style="font-size: 0.75rem; color: #555; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">SYSTEM STATUS</p>
    """, unsafe_allow_html=True)

    # Use Streamlit columns inside sidebar for a mini-stat grid
    # Check model availability
    models_available = check_models_available()
    backend_status = "Active" if any(models_available.values()) else "Not Active"
    backend_color = "#00ff88" if any(models_available.values()) else "#c21717"
    
    s1, s2 = st.columns(2)
    s1.markdown(
        f"""
        <div style="text-align:center; padding:6px;">
            <div style="font-size:12px; color:#9aa0a6;">Backend</div>
            <div style="font-size:14px; color:{backend_color}; font-weight:700; margin-top:4px;">{backend_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    model_count = sum(models_available.values())
    s2.markdown(
        f"""
        <div style="text-align:center; padding:6px;">
            <div style="font-size:12px; color:#9aa0a6;">Models</div>
            <div style="font-size:14px; color:#00ff88; font-weight:700; margin-top:4px;">{model_count}/4 Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Custom status badges
    st.markdown("""
    <div style="background: rgba(30,30,30,0.5); padding: 10px; border-radius: 8px; border: 1px solid #333; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px;">
            <span style="color: #888;">Deep Learning</span>
            <span style="color: #00ff88;">● CNN</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
            <span style="color: #888;">Classifier</span>
            <span style="color: #00ff88;">● SVM / RF</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- FOOTER ---
    st.markdown("""
    <div style="text-align: center; color: #444; font-size: 0.7rem;">
        &copy; 2025 Project Trace Finder<br>
        Academic & Forensic Research
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Hero Section
# -----------------------------------------------------------------------------
st.markdown('<div id="trace-finder"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div style="text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; color: #00d4ff; margin-bottom: 1rem; font-weight: 700;">Source Device Identification</div>
    <h1 class="hero-title">Trace Finder</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Identify the scanner device used to create a document by analyzing unique noise patterns and artifacts.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Feature Cards
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 10px;">📉</div>
        <h3>Extract</h3>
        <p style="font-size: 0.9rem;">Noise Patterns & PRNU</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 10px;">🌊</div>
        <h3>Analyze</h3>
        <p style="font-size: 0.9rem;">FFT & Wavelet Filters</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 2rem; margin-bottom: 10px;">🧠</div>
        <h3>Classify</h3>
        <p style="font-size: 0.9rem;">CNN / SVM Models</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Project Overview 
# -----------------------------------------------------------------------------
st.markdown('<div id="project-overview"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("Project Overview")

col_text, col_details = st.columns([3, 2], gap="large")

with col_text:
    st.subheader("The Objective")
    st.write(
        """
        **Trace Finder** aims to identify the source scanner device (brand/model) by analyzing the unique 
        patterns or artifacts left behind during the scanning process. 
        Each scanner introduces specific noise, texture, or compression traces that are learned 
        by our machine learning models.
        """
    )

    st.subheader("Key Outcomes")
    st.markdown("""
    * **Source Identification:** Understand the concept of source device identification.
    * **Feature Extraction:** Extract scanner-specific features such as noise patterns, frequency domain signals, and artifacts.
    * **Classification:** Train models (CNN, Random Forest, SVM) to distinguish among multiple scanners.
    * **Evaluation:** Visualize feature importance and evaluate model accuracy.
    """)
    
    st.subheader("Use Cases")
    uc1, uc2 = st.columns(2)
    with uc1:
        st.info("**🕵️ Digital Forensics**\n\nDetermine which scanner was used to forge or duplicate legal documents.")
    with uc2:
        st.info("**⚖️ Legal Verification**\n\nEnsure scanned copies submitted in court originated from known and approved devices.")

with col_details:
    st.subheader("Technical Modules")
    # Custom HTML Box for Technical Modules 
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 12px; 
        padding: 20px;
    ">
        <ul style="list-style: none; padding: 0; margin: 0;">
            <li style="margin-bottom: 12px; display: flex; align-items: start;">
                <span style="margin-right: 10px; font-size: 1.1rem;">📂</span>
                <div>
                    <strong style="color: #fff;">1. Data Collection</strong><br>
                    <span style="color: #a0a0a0; font-size: 0.9rem;">Manual scanning & Labeling</span>
                </div>
            </li>
            <li style="margin-bottom: 12px; display: flex; align-items: start;">
                <span style="margin-right: 10px; font-size: 1.1rem;">⚙️</span>
                <div>
                    <strong style="color: #fff;">2. Preprocessing</strong><br>
                    <span style="color: #a0a0a0; font-size: 0.9rem;">Resize, Denoise, Grayscale </span>
                </div>
            </li>
            <li style="margin-bottom: 12px; display: flex; align-items: start;">
                <span style="margin-right: 10px; font-size: 1.1rem;">📉</span>
                <div>
                    <strong style="color: #fff;">3. Feature Extraction</strong><br>
                    <span style="color: #a0a0a0; font-size: 0.9rem;">Wavelet, FFT, PRNU, LBP </span>
                </div>
            </li>
            <li style="margin-bottom: 12px; display: flex; align-items: start;">
                <span style="margin-right: 10px; font-size: 1.1rem;">🧠</span>
                <div>
                    <strong style="color: #fff;">4. Model Training</strong><br>
                    <span style="color: #a0a0a0; font-size: 0.9rem;">CNN, Random Forest, SVM </span>
                </div>
            </li>
            <li style="margin-bottom: 0; display: flex; align-items: start;">
                <span style="margin-right: 10px; font-size: 1.1rem;">✅</span>
                <div>
                    <strong style="color: #fff;">5. Output System</strong><br>
                    <span style="color: #00d4ff; font-weight: 600; font-size: 0.9rem;">Scanner Model + Confidence </span>
                </div>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. System Architecture
# -----------------------------------------------------------------------------
st.markdown('<div id="system-architecture"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("System Architecture")

# Widen the center column so the boxes are not squashed
_, col_main, _ = st.columns([0.5, 6, 0.5])

with col_main:
    graph = """
    digraph G {
        # Layout Settings
        rankdir=LR;
        bgcolor="transparent";
        ranksep=0.4;      # Horizontal spacing between nodes
        splines=ortho;    # Orthogonal lines (straight angles) for a tech look
        
        # Default Node Style - UNIFORMITY
        node [
            shape=box, 
            style="filled,rounded", 
            fontname="Sans-Serif", 
            fontsize=10, 
            width=1.4,        # FIXED WIDTH for all nodes
            height=0.6,       # FIXED HEIGHT for all nodes
            fixedsize=true,   # CRITICAL: Forces all boxes to be same size
            color="#444", 
            fontcolor="#e0e0e0", 
            fillcolor="#1e1e24", 
            penwidth=1.5
        ];
        
        # Edge Style
        edge [color="#00d4ff", penwidth=1.5, arrowhead=vee, arrowsize=0.8];

        # Define Nodes
        Input      [label="📄 Scanned\\nImage Input", color="#4E7B89", fontcolor="#fff"];
        Preprocess [label="⚙️ Image\\nPreprocessing"];
        Feature    [label="📉 Feature\\nExtraction"];
        Model      [label="🧠 ML/DL Model\\n(Classifier)", fillcolor="#003344", color="#004e66", fontcolor="#fff"];
        Output     [label="✅ Prediction\\nOutput", shape=box, fillcolor="#025B79", fontcolor="#000", color="#028BB8"];

        # Connections
        Input -> Preprocess;
        Preprocess -> Feature;
        Feature -> Model;
        Model -> Output;
    }
    """
    st.graphviz_chart(graph, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. Live Demo
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div id="forensic-analysis-lab"></div>', unsafe_allow_html=True)

# Using Native Streamlit Container for stability
with st.container(border=True):
    st.markdown('<h2 style="text-align: center; margin-top: 0;">🧪 Forensic Analysis Lab</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Upload a scanned document to detect source device artifacts.</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("### 1. Input Source")
        uploaded_file = st.file_uploader("Drop scanned image", type=['jpg', 'png', 'tif'])
        
        # Sample Images Feature
        st.markdown("##### Or try a sample:")
        sample_options = {
            "None": None,
            "📷 EpsonV550 ": "samples/EpsonV550_sample.jpg",
            "📷 HP": "samples/HP_sample.jpg",
            "📷 EpsonV370-1 ": "samples/EpsonV370-1_sample.jpg",
            "📷 Canon120-1 ": "samples/Canon120-1_sample.jpg"
        }
        selected_sample = st.selectbox("Select a sample image", list(sample_options.keys()), label_visibility="collapsed")
        
        st.markdown("### 2. Select Model")
        feature_mode = st.radio("Choose Analysis Method", [
            "🧠 Hybrid CNN (Residual + Features)",
            "🔬 Standalone CNN (Deep Learning)",
            "🌲 Random Forest (Baseline)",
            "⚡ SVM (Baseline)",
            "🔍 Explainability (Grad-CAM)"
        ])
        
        st.write("")  # Spacer
        analyze_btn = st.button("🚀 Identify Scanner", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 3. Analysis Results")
        
        # Determine image source (uploaded or sample)
        image_bytes = None
        file_size = 0
        image_source = None
        
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            file_size = len(image_bytes)
            image_source = "upload"
        elif selected_sample != "None" and sample_options[selected_sample]:
            sample_path = sample_options[selected_sample]
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                file_size = len(image_bytes)
                image_source = selected_sample
        
        if image_bytes and analyze_btn:
            with st.status("Executing Trace Finder Pipeline...", expanded=True) as status:
                if image_source != "upload":
                    st.write(f"📁 Using sample: {image_source}")
                st.write("📥 Loading image...")
                time.sleep(0.2)
                
                try:
                    scanner_name = None
                    confidence = 0.0
                    all_probs = {}
                    gradcam_img = None
                    
                    if "Hybrid CNN" in feature_mode:
                        st.write("🔄 Computing residual (wavelet denoising)...")
                        st.write("📊 Extracting FFT + LBP features...")
                        st.write("🧠 Running Hybrid CNN classifier...")
                        scanner_name, confidence, all_probs = predict_hybrid_cnn(image_bytes)
                        
                    elif "Standalone CNN" in feature_mode:
                        st.write("🖼️ Preprocessing image (128x128 RGB)...")
                        st.write("🔬 Running CNN classifier...")
                        scanner_name, confidence, all_probs = predict_cnn(image_bytes)
                        
                    elif "Random Forest" in feature_mode:
                        st.write("📊 Extracting metadata features...")
                        st.write("🌲 Running Random Forest classifier...")
                        scanner_name, confidence, all_probs = predict_baseline(image_bytes, file_size, 'rf')
                        
                    elif "SVM" in feature_mode:
                        st.write("📊 Extracting metadata features...")
                        st.write("⚡ Running SVM classifier...")
                        scanner_name, confidence, all_probs = predict_baseline(image_bytes, file_size, 'svm')
                        
                    elif "Explainability" in feature_mode:
                        st.write("🔄 Computing residual...")
                        st.write("🔍 Generating Grad-CAM heatmap...")
                        gradcam_img, scanner_name, confidence = generate_gradcam(image_bytes)
                    
                    status.update(label="✅ Identification Complete", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="❌ Error", state="error", expanded=True)
                    st.error(f"Model inference failed: {str(e)}")
                    scanner_name = None
            
            if scanner_name:
                st.markdown("---")
                
                r1, r2 = st.columns(2)
                with r1:
                    st.caption("PREDICTED SOURCE SCANNER")
                    st.markdown(f"<h2 style='color:#fff !important; margin:0; font-size: 2rem !important;'>{scanner_name}</h2>", unsafe_allow_html=True)
                
                with r2:
                    delta_text = "✓ Above target" if confidence >= 85 else "Below 85% target"
                    st.metric("Confidence Score", f"{confidence:.1f}%", delta_text)
                
                # Show Grad-CAM if in explainability mode
                if gradcam_img is not None:
                    st.caption("GRAD-CAM VISUALIZATION")
                    st.image(gradcam_img, caption="Regions the model focused on", use_container_width=True)
                
                # Show top predictions
                if all_probs:
                    st.caption("TOP PREDICTIONS")
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    # Create a DataFrame for correct plotting with single color
                    prob_df = pd.DataFrame(sorted_probs, columns=["Scanner", "Probability"])
                    prob_df.set_index("Scanner", inplace=True)
                    st.bar_chart(prob_df, color="#00d4ff")
                
                st.success(f"🎯 Outcome: Document identified as scanned by **{scanner_name}** device.")

        elif not uploaded_file:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; border: 2px dashed #444; border-radius: 12px; color: #666; margin-top: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                <div style="font-weight: 600; color: #888;">Waiting for scanned image...</div>
                <small>Upload a file to extract source traces.</small>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. Footer
# -----------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; opacity: 0.5; font-size: 0.8rem;">
        Trace Finder | Forensic Scanner Identification Project<br>
        &copy; 2025 Machine Learning Module
    </div>
    """, 
    unsafe_allow_html=True
)