from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from config import APP_TITLE
from ui_design_system.streamlit_style import apply_app_style
from panels import (
    render_panel_1_upload,
    render_panel_2_compression,
    render_panel_3_encoding,
    render_panel_4_experiment,
    render_panel_5_decoding,
    render_panel_6_analysis,
)
from text_dna_unified_panel import render_text_dna_storage_panel
from audio_dna_tab import render_audio_dna_storage_panel
from video_dna_tab import render_video_dna_storage_panel


APP_BUILD_ID = "v15_original_algorithms_clean_ui"

APP_STEPS = [
    (1, "Input"),
    (2, "Compression / Method"),
    (3, "DNA Encoding"),
    (4, "Strand / NGS"),
    (5, "Decode / Repair"),
    (6, "Analysis"),
]

DOMAIN_TABS = [
    ("image", "Image"),
    ("text", "Text"),
    ("audio", "Audio"),
    ("video", "Video"),
]


_ORIGINAL_INFO = st.info
_ORIGINAL_SUCCESS = st.success


def _compact_note(message: str, *, kind: str = "neutral") -> None:
    """Render non-critical messages without blue/green Streamlit alert blocks."""
    text = escape(str(message or ""))
    st.markdown(f'<div class="compact-note compact-note-{kind}">{text}</div>', unsafe_allow_html=True)


def _patch_streamlit_status_messages() -> None:
    """Keep errors/warnings visible, but remove blue/green explanatory blocks."""
    st.info = lambda body, *args, **kwargs: _compact_note(body, kind="neutral")  # type: ignore[assignment]
    st.success = lambda body, *args, **kwargs: _compact_note(body, kind="done")  # type: ignore[assignment]


def _has_meaningful_value(key: str) -> bool:
    value = st.session_state.get(key)
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray, str, list, tuple, dict, set)):
        return len(value) > 0
    try:
        return bool(value)
    except Exception:
        return True


def _any_keys(keys: list[str]) -> bool:
    return any(_has_meaningful_value(k) for k in keys)


def _pipeline_checks(domain: str) -> dict[int, bool]:
    """Status only reads native pipeline states; algorithms remain unchanged."""
    if domain == "image":
        decoded_ok = _any_keys(["decoded_data", "restored_info", "raw_restore_info"])
        return {
            1: _any_keys(["input_bytes", "input_path"]),
            2: _has_meaningful_value("stored_bytes"),
            3: _has_meaningful_value("dna"),
            4: _has_meaningful_value("strand_rows"),
            5: decoded_ok or bool(st.session_state.get("decode_error")),
            6: decoded_ok,
        }
    if domain == "text":
        return {
            1: _has_meaningful_value("text_input_value"),
            2: _has_meaningful_value("text_selected_package"),
            3: _has_meaningful_value("text_dna"),
            4: _has_meaningful_value("text_strand_rows"),
            5: _any_keys(["text_final_text", "text_decoded_text", "text_repaired_text"]) or bool(st.session_state.get("text_decode_error")),
            6: _has_meaningful_value("text_validation"),
        }
    if domain == "audio":
        return {
            1: _has_meaningful_value("audio_input_bytes"),
            2: _has_meaningful_value("audio_payload"),
            3: _has_meaningful_value("audio_dna"),
            4: _has_meaningful_value("audio_strand_rows"),
            5: _has_meaningful_value("audio_recovered_wav") or bool(st.session_state.get("audio_decode_error")),
            6: _has_meaningful_value("audio_recovery_metrics"),
        }
    if domain == "video":
        return {
            1: _any_keys(["video_input_bytes", "video_input_path"]),
            2: _has_meaningful_value("video_payload"),
            3: _has_meaningful_value("video_dna"),
            4: _has_meaningful_value("video_strand_rows"),
            5: _has_meaningful_value("video_reconstructed_result") or bool(st.session_state.get("video_decode_error")),
            6: _any_keys(["video_validation_metrics", "video_reconstruction_metrics"]),
        }
    return {i: False for i in range(1, 7)}


def _step_state(step_no: int, domain: str) -> tuple[str, str]:
    checks = _pipeline_checks(domain)
    review_keys = {
        "image": "decode_error",
        "text": "text_decode_error",
        "audio": "audio_decode_error",
        "video": "video_decode_error",
    }
    if step_no == 5 and st.session_state.get(review_keys.get(domain, "")):
        return "review", "Review"
    if checks.get(step_no, False):
        return "done", "Done"
    previous_done = all(checks.get(i, False) for i in range(1, step_no))
    if previous_done:
        return "current", "Next"
    return "waiting", "Waiting"

def _render_global_style() -> None:
    st.markdown(
        f"""
<style>
/* build: {APP_BUILD_ID} */

:root {{
  --bg: #F6F8FB;
  --surface: #FFFFFF;
  --surface-soft: #EEF4F8;
  --border: #D6E0EA;
  --text: #102033;
  --muted: #5F6F82;

  --primary: #0B5CAD;
  --primary-soft: #DCEEFF;
  --primary-border: #8BC4FF;

  --success: #0E9F6E;
  --success-soft: #DDF7EE;
  --success-border: #74D9B2;

  --warning: #C27803;
  --warning-soft: #FFF4D6;
  --warning-border: #F6C453;

  --danger: #D92D20;
  --danger-soft: #FEE4E2;

  --shadow: rgba(16, 32, 51, 0.12);
}}

.stApp {{
  background: var(--bg);
  color: var(--text);
}}

.block-container {{
  padding-top: 0.85rem;
  padding-bottom: 2.2rem;
  max-width: 1380px;
}}

h1, h2, h3, h4 {{
  letter-spacing: -0.02em;
  color: var(--text);
}}

.app-title {{
  font-size: 1.35rem;
  font-weight: 850;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0.1rem 0 0.75rem 0;
}}

.pipeline-status-sticky {{
  position: fixed;
  top: 0.55rem;
  left: 50%;
  transform: translateX(-50%);
  width: min(1320px, calc(100vw - 3rem));
  z-index: 999999;
  background: rgba(255, 255, 255, 0.97);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 0.65rem;
  box-shadow: 0 10px 28px var(--shadow);
}}

.pipeline-status-spacer {{
  height: 10px;
}}

.pipeline-steps {{
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.48rem;
}}

.pipeline-step {{
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.55rem 0.62rem;
  background: var(--surface-soft);
  min-height: 58px;
}}

.step-main {{
  display: flex;
  align-items: center;
  gap: 0.42rem;
  min-width: 0;
}}

.step-num {{
  width: 23px;
  height: 23px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.78rem;
  background: #D8E2EE;
  color: var(--text);
  flex: 0 0 auto;
}}

.step-name {{
  font-weight: 760;
  font-size: 0.82rem;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.step-state {{
  margin-top: 0.32rem;
  font-size: 0.75rem;
  font-weight: 800;
  color: var(--muted);
}}

.pipeline-step.done {{
  background: var(--success-soft);
  border-color: var(--success-border);
}}

.pipeline-step.done .step-num {{
  background: var(--success);
  color: white;
}}

.pipeline-step.done .step-state {{
  color: #08785A;
}}

.pipeline-step.current {{
  background: var(--primary-soft);
  border-color: var(--primary-border);
}}

.pipeline-step.current .step-num {{
  background: var(--primary);
  color: white;
}}

.pipeline-step.current .step-state {{
  color: var(--primary);
}}

.pipeline-step.waiting {{
  opacity: 0.82;
}}

.pipeline-step.review {{
  background: var(--warning-soft);
  border-color: var(--warning-border);
}}

.pipeline-step.review .step-num {{
  background: var(--warning);
  color: white;
}}

.pipeline-step.review .step-state {{
  color: #8A5A00;
}}

.compact-note {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.58rem 0.72rem;
  color: var(--muted);
  background: var(--surface-soft);
  font-size: 0.88rem;
  margin: 0.35rem 0;
}}

.compact-note-done {{
  color: #08785A;
  background: var(--success-soft);
  border-color: var(--success-border);
}}

div[data-testid="stMetric"] {{
  border: 1px solid var(--border);
  border-radius: 15px;
  padding: 0.62rem 0.72rem;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 4px 14px rgba(16, 32, 51, 0.035);
}}

[data-testid="stImage"] img {{
  max-width: 100%;
  max-height: 430px;
  object-fit: contain;
}}

[data-testid="stVideo"] video {{
  width: 100% !important;
  max-height: 430px !important;
  object-fit: contain;
  border-radius: 12px;
}}

[data-testid="stAudio"] audio {{
  width: 100% !important;
}}

@media (max-width: 1100px) {{
  .pipeline-status-sticky {{
    width: calc(100vw - 2rem);
  }}

  .pipeline-steps {{
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }}

  .pipeline-status-spacer {{
    height: 10px;
  }}
}}

@media (max-width: 700px) {{
  .pipeline-status-sticky {{
    width: calc(100vw - 1rem);
    top: 0.35rem;
  }}

  .pipeline-steps {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .pipeline-status-spacer {{
    height: 10px;
  }}
}}
</style>
""",
        unsafe_allow_html=True,
    )
# def _render_global_style() -> None:
#     st.markdown(
#         f"""
# <style>
# /* build: {APP_BUILD_ID} */
# .block-container {{
#   padding-top: 0.85rem;
#   padding-bottom: 2.2rem;
#   max-width: 1380px;
# }}
# h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
# .app-title {{
#   font-size: 1.35rem;
#   font-weight: 850;
#   letter-spacing: -0.03em;
#   color: #0f172a;
#   margin: 0.1rem 0 0.75rem 0;
# }}
# .pipeline-status-sticky {{
#   position: fixed;
#   top: 0.55rem;
#   left: 50%;
#   transform: translateX(-50%);
#   width: min(1320px, calc(100vw - 3rem));
#   z-index: 999999;
#   background: rgba(255, 255, 255, 0.97);
#   backdrop-filter: blur(12px);
#   border: 1px solid #e5e7eb;
#   border-radius: 18px;
#   padding: 0.65rem;
#   box-shadow: 0 10px 28px rgba(15, 23, 42, 0.13);
# }}
# .pipeline-status-spacer {{ height: 10px; }}
# .pipeline-steps {{
#   display: grid;
#   grid-template-columns: repeat(6, minmax(0, 1fr));
#   gap: 0.48rem;
# }}
# .pipeline-step {{
#   border: 1px solid #e5e7eb;
#   border-radius: 14px;
#   padding: 0.55rem 0.62rem;
#   background: #f8fafc;
#   min-height: 58px;
# }}
# .step-main {{ display: flex; align-items: center; gap: 0.42rem; min-width: 0; }}
# .step-num {{
#   width: 23px; height: 23px; border-radius: 999px;
#   display: inline-flex; align-items: center; justify-content: center;
#   font-weight: 800; font-size: 0.78rem; background: #e2e8f0; color: #334155;
#   flex: 0 0 auto;
# }}
# .step-name {{
#   font-weight: 760; font-size: 0.82rem; color: #0f172a;
#   white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
# }}
# .step-state {{ margin-top: 0.32rem; font-size: 0.75rem; font-weight: 800; color: #64748b; }}
# .pipeline-step.done {{ background: #ecfdf5; border-color: #86efac; }}
# .pipeline-step.done .step-num {{ background: #22c55e; color: white; }}
# .pipeline-step.done .step-state {{ color: #15803d; }}
# .pipeline-step.current {{ background: #eff6ff; border-color: #93c5fd; }}
# .pipeline-step.current .step-num {{ background: #2563eb; color: white; }}
# .pipeline-step.current .step-state {{ color: #1d4ed8; }}
# .pipeline-step.waiting {{ opacity: 0.82; }}
# .pipeline-step.review {{ background: #fff7ed; border-color: #fdba74; }}
# .pipeline-step.review .step-num {{ background: #f97316; color: white; }}
# .pipeline-step.review .step-state {{ color: #c2410c; }}
# .compact-note {{
#   border: 1px solid #e5e7eb;
#   border-radius: 12px;
#   padding: 0.58rem 0.72rem;
#   color: #475569;
#   background: #f8fafc;
#   font-size: 0.88rem;
#   margin: 0.35rem 0;
# }}
# .compact-note-done {{ color: #166534; background: #f7fef9; border-color: #dcfce7; }}
# div[data-testid="stMetric"] {{
#   border: 1px solid rgba(125, 125, 125, 0.14);
#   border-radius: 15px;
#   padding: 0.62rem 0.72rem;
#   background: rgba(125, 125, 125, 0.035);
# }}
# [data-testid="stImage"] img {{
#   max-width: 100%;
#   max-height: 430px;
#   object-fit: contain;
# }}
# [data-testid="stVideo"] video {{
#   width: 100% !important;
#   max-height: 430px !important;
#   object-fit: contain;
#   border-radius: 12px;
# }}
# [data-testid="stAudio"] audio {{ width: 100% !important; }}
# @media (max-width: 1100px) {{
#   .pipeline-status-sticky {{ width: calc(100vw - 2rem); }}
#   .pipeline-steps {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
#   .pipeline-status-spacer {{ height: 10px; }}
# }}
# @media (max-width: 700px) {{
#   .pipeline-status-sticky {{ width: calc(100vw - 1rem); top: 0.35rem; }}
#   .pipeline-steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
#   .pipeline-status-spacer {{ height: 10px; }}
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


def _stepper_html(domain: str) -> str:
    parts = ['<div class="pipeline-status-sticky"><div class="pipeline-steps">']
    for number, label in APP_STEPS:
        css_class, state_text = _step_state(number, domain)
        parts.append(
            f"""
<div class="pipeline-step {css_class}">
  <div class="step-main">
    <span class="step-num">{number}</span>
    <span class="step-name">{label}</span>
  </div>
  <div class="step-state">{state_text}</div>
</div>
"""
        )
    parts.append("</div></div><div class='pipeline-status-spacer'></div>")
    return "".join(parts)


def _render_stepper(domain: str, target: Any | None = None) -> None:
    html = _stepper_html(domain)
    if target is None:
        st.markdown(html, unsafe_allow_html=True)
    else:
        target.markdown(html, unsafe_allow_html=True)



def render_image_pipeline() -> None:
    st.session_state["active_domain"] = "image"
    status_placeholder = st.empty()
    _render_stepper("image", status_placeholder)
    render_panel_1_upload()
    render_panel_2_compression()
    render_panel_3_encoding()
    render_panel_4_experiment()
    render_panel_5_decoding()
    render_panel_6_analysis()
    _render_stepper("image", status_placeholder)


def render_text_pipeline() -> None:
    status_placeholder = st.empty()
    _render_stepper("text", status_placeholder)
    render_text_dna_storage_panel()
    _render_stepper("text", status_placeholder)


def render_audio_pipeline() -> None:
    status_placeholder = st.empty()
    _render_stepper("audio", status_placeholder)
    render_audio_dna_storage_panel()
    _render_stepper("audio", status_placeholder)


def render_video_pipeline() -> None:
    status_placeholder = st.empty()
    _render_stepper("video", status_placeholder)
    render_video_dna_storage_panel()
    _render_stepper("video", status_placeholder)


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    apply_app_style()
    _render_global_style()
    _patch_streamlit_status_messages()
    st.markdown('<div class="app-title"><br/>DNA Data Storage System</div>', unsafe_allow_html=True)

    tabs = st.tabs([label for _, label in DOMAIN_TABS])
    with tabs[0]:
        render_image_pipeline()
    with tabs[1]:
        render_text_pipeline()
    with tabs[2]:
        render_audio_pipeline()
    with tabs[3]:
        render_video_pipeline()


if __name__ == "__main__":
    render_app()
