import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import io
import tempfile
import os
import pandas as pd

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.recommendation.severity_estimator import SeverityEstimator
from src.recommendation.engine import RuleBasedEngine
from src.recommendation.candidate_generator import CandidateGenerator
from src.pipeline.adaptive_selector import AdaptiveSelector
from src.evaluation.quality import evaluate_no_reference_quality
from src.enhancement.library import (
    apply_gamma_correction, apply_clahe,
    apply_denoising, apply_unsharp_mask, apply_exposure_correction
)

# -- Page config --
st.set_page_config(
    page_title="Intelligent Image Enhancer",
    layout="wide"
)

# -- Load models (cached so they only load once) --
@st.cache_resource
def load_models():
    estimator = SeverityEstimator()
    selector  = AdaptiveSelector()
    return estimator, selector

estimator, selector = load_models()

CLASSES = ['low_brightness', 'low_contrast', 'noise', 'blur', 'overexposure']
LABELS  = ['Low Brightness', 'Low Contrast', 'Noise', 'Blur', 'Overexposure']

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# -- Header --
st.title("Intelligent Image Enhancer")
st.markdown("*Adaptive Quality-Aware Enhancement using Multi-Label CNN*")
st.divider()

# -- Sidebar: Upload --
with st.sidebar:
    st.header("Upload Image")
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.divider()
    st.markdown("**About**")
    st.caption(
        "This system uses a MobileNetV2 CNN to detect up to 5 degradations, "
        "then adaptively selects the best enhancement pipeline."
    )

if uploaded is None:
    st.info("Upload an image using the sidebar to get started.")
    st.stop()

# -- Save upload to a temp file (needed by OpenCV) --
suffix = Path(uploaded.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = Path(tmp.name)

original_pil = Image.open(tmp_path).convert("RGB")
original_cv2 = pil_to_cv2(original_pil)

# -- SECTION 1: AI Quality Analysis --
st.subheader("AI Quality Analysis")

with st.spinner("Running CNN degradation detection..."):
    severity_profile = estimator.estimate(tmp_path)

col_img, col_bars = st.columns([1, 1])

with col_img:
    st.image(original_pil, caption="Original Image", use_container_width=True)

with col_bars:
    st.markdown("**Detected Degradation Severity**")
    for cls, label in zip(CLASSES, LABELS):
        score = severity_profile[cls]
        st.markdown(f"`{label}`")
        st.progress(score, text=f"{score*100:.1f}%")

st.divider()

# -- SECTION 2: Recommended Pipeline --
st.subheader("AI Recommended Pipeline")

with st.spinner("Generating and evaluating candidate pipelines..."):
    best_img_cv2, best_name, results = selector.process_and_select(tmp_path, severity_profile)

best_pil = cv2_to_pil(best_img_cv2)
best_result = results[best_name]

col_a, col_b = st.columns(2)
with col_a:
    st.image(original_pil, caption="Original", use_container_width=True)
with col_b:
    st.image(best_pil, caption=f"AI Enhanced ({best_name})", use_container_width=True)

st.success(f"Best Pipeline: **{best_result['pipeline']}**  |  Score: `{best_result['ebs_score']}`")

# Candidate comparison table
st.markdown("**Candidate Pipeline Comparison**")
rows = []
for name, info in results.items():
    rows.append({
        "Strategy":   name,
        "Pipeline":   info['pipeline'],
        "Score (EBS)":info['ebs_score'],
        "Sharpness":  info['sharpness'],
        "Contrast":   info['contrast'],
        "Artifacts":  info['artifacts'],
        "Winner":     "YES" if name == best_name else ""
    })
st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

st.divider()

# -- SECTION 3: Manual Customization --
st.subheader("Manual Customization")

c1, c2 = st.columns(2)
with c1:
    do_denoise  = st.checkbox("Apply Denoising",        value=(severity_profile['noise'] > 0.5))
    do_gamma    = st.checkbox("Apply Gamma Correction", value=(severity_profile['low_brightness'] > 0.5))
    do_clahe    = st.checkbox("Apply CLAHE (Contrast)", value=(severity_profile['low_contrast'] > 0.5))
    do_sharpen  = st.checkbox("Apply Sharpening",       value=(severity_profile['blur'] > 0.5))
    do_exposure = st.checkbox("Apply Exposure Fix",     value=(severity_profile['overexposure'] > 0.5))

with c2:
    gamma_val   = st.slider("Gamma",            0.2, 1.5, 0.7, 0.05)
    sharpen_val = st.slider("Sharpness",        0.5, 4.0, 1.5, 0.1)
    denoise_d   = st.slider("Denoise Radius",   3,   15,  9,   2)
    clahe_clip  = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5)

if st.button("Apply Custom Enhancement", type="primary"):
    custom_img = original_cv2.copy()
    steps_applied = []

    if do_denoise:
        custom_img = apply_denoising(custom_img, d=denoise_d)
        steps_applied.append("Denoise")
    if do_exposure:
        custom_img = apply_exposure_correction(custom_img)
        steps_applied.append("Exposure Fix")
    if do_gamma:
        custom_img = apply_gamma_correction(custom_img, gamma=gamma_val)
        steps_applied.append("Gamma")
    if do_clahe:
        custom_img = apply_clahe(custom_img, clip_limit=clahe_clip)
        steps_applied.append("CLAHE")
    if do_sharpen:
        custom_img = apply_unsharp_mask(custom_img, amount=sharpen_val)
        steps_applied.append("Sharpen")

    custom_pil = cv2_to_pil(custom_img)
    caption = " -> ".join(steps_applied) if steps_applied else "None"
    st.image(custom_pil, caption=f"Custom: {caption}", use_container_width=True)

    ebs, sharp, contrast, arts = evaluate_no_reference_quality(custom_img)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("EBS Score",  f"{ebs:.3f}")
    col_m2.metric("Sharpness",  f"{sharp:.3f}")
    col_m3.metric("Contrast",   f"{contrast:.3f}")
    col_m4.metric("Artifacts",  f"{arts:.3f}")

    buf = io.BytesIO()
    custom_pil.save(buf, format="PNG")
    st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced.png", "image/png")

# -- Cleanup --
try:
    os.unlink(tmp_path)
except Exception:
    pass
