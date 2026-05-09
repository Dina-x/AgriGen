from io import BytesIO
from pathlib import Path
import streamlit as st
from model_utils import generate_image, load_lora_pipeline

st.set_page_config(page_title="AgriGen", page_icon="🌱", layout="wide")

# -----------------------------
# Theme Toggle
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.session_state.theme

if theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#ffffff"
    muted_text = "#d9e2d5"
    input_bg = "rgba(145, 210, 110, 0.08)"
    border_color = "rgba(145, 210, 110, 0.45)"
    result_bg = "rgba(145, 210, 110, 0.10)"
    button_bg = "#65b741"
    button_bg2 = "#9bd65c"
    download_bg = "#171b24"
    download_text = "#ffffff"
    top_bar = "#0e1117"
    toggle_bg = "#202633"
else:
    bg_color = "#fbfff7"
    text_color = "#314032"
    muted_text = "#5f6f60"
    input_bg = "#ffffff"
    border_color = "rgba(126, 172, 100, 0.32)"
    result_bg = "#ffffff"
    button_bg = "#8ccf62"
    button_bg2 = "#b9e48f"
    download_bg = "#ffffff"
    download_text = "#314032"
    top_bar = "#fbfff7"
    toggle_bg = "#eaf5df"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

header[data-testid="stHeader"] {{
    background-color: {top_bar} !important;
}}

.block-container {{
    max-width: 1050px;
    padding-top: 1.2rem;
}}

.app-title {{
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-size: 58px;
    font-weight: 800;
    margin-bottom: 28px;
    color: {text_color};
}}

.result-box {{
    background: {result_bg};
    border: 1px dashed {border_color};
    border-radius: 22px;
    min-height: 320px;
    padding: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: {muted_text};
}}

.stTextArea > div > div {{
    background: {input_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 18px !important;
    box-shadow: none !important;
}}

textarea {{
    background: transparent !important;
    color: {text_color} !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 18px !important;
    font-size: 15px !important;
    padding: 14px !important;
}}

.stButton > button {{
    background: linear-gradient(90deg, {button_bg}, {button_bg2});
    color: white !important;
    border: none !important;
    border-radius: 16px;
    height: 50px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 16px;
}}

.stDownloadButton > button {{
    background: {download_bg} !important;
    color: {download_text} !important;
    border: 1px solid {border_color} !important;
    border-radius: 16px;
    height: 46px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
}}

.stDownloadButton > button:hover {{
    border-color: {button_bg} !important;
}}

div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child button[kind="secondary"] {{
    background: {toggle_bg} !important;
    color: {text_color} !important;
    border: 1px solid {border_color} !important;
    border-radius: 16px !important;
    height: 44px !important;
    font-size: 20px !important;
}}

h1, h2, h3 {{
    color: {text_color} !important;
    font-family: 'Poppins', sans-serif !important;
}}

label, p, span {{
    color: {muted_text} !important;
}}

.stMarkdown, .stRadio, .stExpander {{
    color: {text_color} !important;
}}

.stAlert {{
    border-radius: 16px;
}}

[data-testid="stToolbar"] {{
    display: none !important;
}}

#MainMenu, footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([9, 1])
with top_right:
    icon = "☀️" if st.session_state.theme == "Dark" else "🌙"
    if st.button(icon, key="theme_toggle", use_container_width=True):
        st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"
        st.rerun()

@st.cache_resource
def load_resources():
    return load_lora_pipeline("models")

def image_to_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

st.markdown("<div class='app-title'>🌱 AgriGen</div>", unsafe_allow_html=True)

if not Path("models").exists():
    st.error("The models folder was not found. Please place the AgriGen_Lite_Fast files inside a folder named models.")
    st.stop()

with st.spinner("Loading AgriGen model... this may take a while on first run."):
    try:
        pipe, supported_prompts = load_resources()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### ✨ Describe your image")

    prompt = st.text_area(
        "Prompt",
        placeholder="Example: fresh red apple",
        height=150,
        label_visibility="collapsed"
    )

    style = st.radio(
        "Visual style",
        ["Natural", "Bright", "Soft grayscale", "Sketch"],
        horizontal=True
    )

    generate_btn = st.button("Generate Image ✨", use_container_width=True)

    with st.expander("About AgriGen"):
        st.write("""
        AgriGen is an AI-based image generation prototype designed for fruits,
        vegetables, and agricultural-related visuals. It uses a lightweight LoRA
        fine-tuning approach on top of a text-to-image model.
        """)

with right:
    st.markdown("### 🖼️ Generated Result")

    if generate_btn:
        if not prompt.strip():
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Generating your image..."):
                try:
                    img, matched_class = generate_image(
                        prompt=prompt,
                        pipe=pipe,
                        supported_prompts=supported_prompts,
                        style=style,
                        seed=None,
                    )
                except Exception as e:
                    st.warning(str(e))
                    st.stop()

            st.image(img, caption=f"Prompt: {prompt}", width=380)

            st.download_button(
                "Download Image",
                data=image_to_bytes(img),
                file_name="agrigen_output.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.markdown(
            "<div class='result-box'>Your generated image will appear here.</div>",
            unsafe_allow_html=True
        )
