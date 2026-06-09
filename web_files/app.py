from io import BytesIO
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="AgriGen", page_icon="🌱", layout="wide")

# -------------------------------------------------------------
# Backend API URL
# -------------------------------------------------------------
# Enter the ngrok public URL that exposes the FastAPI backend
# Example: https://abcd1234.ngrok-free.app
API_URL = ""

# -------------------------------------------------------------
# Theme Toggle Setup
# -------------------------------------------------------------
# Fixed: Set "Light" as the default theme instead of Dark
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme = st.session_state.theme

# -------------------------------------------------------------
# Theme Colors
# -------------------------------------------------------------
if theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#ffffff"
    muted_text = "#d9e2d5"
    input_bg = "#f4f8f2"
    prompt_text_color = "#000000"
    border_color = "rgba(145, 210, 110, 0.45)"
    result_bg = "rgba(145, 210, 110, 0.10)"
    button_bg = "#65b741"
    button_bg2 = "#9bd65c"
    download_bg = "#171b24"
    download_text = "#ffffff"
    top_bar = "#0e1117"
    toggle_bg = "#65b741"
    expander_color = "#d9e2d5"
else:
    bg_color = "#fbfff7"
    text_color = "#314032"
    muted_text = "#5f6f60"
    input_bg = "#ffffff"
    prompt_text_color = "#000000"
    border_color = "rgba(126, 172, 100, 0.32)"
    result_bg = "#ffffff"
    button_bg = "#8ccf62"
    button_bg2 = "#b9e48f"
    download_bg = "#ffffff"
    download_text = "#314032"
    top_bar = "#fbfff7"
    toggle_bg = "#8ccf62"
    expander_color = "#314032"

# -------------------------------------------------------------
# Custom CSS Styling
# -------------------------------------------------------------
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
    position: relative !important;
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
    color: {prompt_text_color} !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 18px !important;
    font-size: 15px !important;
    padding: 14px !important;
}}

textarea::placeholder {{
    color: #6f766e !important;
}}

.stButton > button:not([key="theme_toggle"]) {{
    background: linear-gradient(90deg, {button_bg}, {button_bg2});
    color: white !important;
    border: none !important;
    border-radius: 16px;
    height: 50px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 16px;
}}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {{
    color: white !important;
    border: none !important;
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

/* Unclipped Absolute Top-Right Floating Theme Toggle Fixed Layout */
.floating-toggle-container {{
    position: absolute !important;
    top: 5px !important;
    right: 0px !important;
    z-index: 999999 !important;
}}

/* Enforcing clear display block metrics to prevent any hidden parent cuts */
div[data-testid="element-container"]:has(button[key="theme_toggle"]) {{
    overflow: visible !important;
}}

.stButton > button[key="theme_toggle"] {{
    background: {toggle_bg} !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    width: 58px !important;
    min-width: 58px !important;
    height: 48px !important;
    font-size: 22px !important;
    padding: 0px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1) !important;
    overflow: visible !important;
}}

h1, h2, h3 {{
    color: {text_color} !important;
    font-family: 'Poppins', sans-serif !important;
}}

label, p, span {{
    color: {muted_text} !important;
}}

.stMarkdown,
.stRadio,
.stExpander {{
    color: {text_color} !important;
}}

.stAlert {{
    border-radius: 16px;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active {{
    color: {expander_color} !important;
    background-color: transparent !important;
    font-weight: 600 !important;
}}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:focus p,
[data-testid="stExpander"] summary:active p {{
    color: {expander_color} !important;
}}

[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary:hover svg,
[data-testid="stExpander"] summary:focus svg,
[data-testid="stExpander"] summary:active svg {{
    color: {expander_color} !important;
    fill: {expander_color} !important;
}}

[data-testid="stToolbar"] {{
    display: none !important;
}}

#MainMenu,
footer {{
    visibility: hidden;
}}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Absolute Floating Top-Right Theme Toggle Button Row
# -------------------------------------------------------------
st.markdown('<div class="floating-toggle-container">', unsafe_allow_html=True)
icon = "☀️" if st.session_state.theme == "Dark" else "🌙"
if st.button(icon, key="theme_toggle"):
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Convert generated image to bytes for download
# -------------------------------------------------------------
def image_to_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

# -------------------------------------------------------------
# App Title
# -------------------------------------------------------------
st.markdown("<div class='app-title'>🌱 AgriGen</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

# -------------------------------------------------------------
# Left Column: Prompt Input and Options
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Right Column: Generated Result
# -------------------------------------------------------------
with right:
    st.markdown("### 🖼️ Generated Result")

    if generate_btn:
        if not prompt.strip():
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Generating your image..."):
                try:
                    response = requests.post(
                        f"{API_URL}/generate",
                        json={"prompt": prompt, "style": style},
                        timeout=120
                    )

                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))

                        st.image(img, caption=f"Prompt: {prompt}", width=380)

                        st.download_button(
                            "Download Image",
                            data=image_to_bytes(img),
                            file_name="agrigen_output.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        error_detail = response.json().get("detail", "Generation Error")
                        st.error(f"Error from server: {error_detail}")

                except requests.exceptions.RequestException as e:
                    st.error(
                        f"Could not connect to Colab backend. "
                        f"Please check if Colab is running. Error: {e}"
                    )
    else:
        st.markdown(
            "<div class='result-box'>Your generated image will appear here.</div>",
            unsafe_allow_html=True
        )