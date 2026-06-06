# ==========================================================
# VIDEO QA SYSTEM - DESKTOP APP (NiceGUI, native window)
# ==========================================================
# Runs as a standalone desktop window (via pywebview). Reuses all
# model/inference logic from app_core.py. The heavy model is loaded
# ONCE and kept resident in memory.
#
# Run:  python desktop_app.py
# ==========================================================

import os
import tempfile
from datetime import datetime

from nicegui import ui, run, app

import app_core as core

# Global app state
STATE = {
    "models": None,
    "context": None,
    "current_video_name": None,
}


# ==========================================================
# MODEL LOADING (background thread, keeps UI responsive)
# ==========================================================

async def load_models_async(status_label, container):
    def progress(msg):
        # Update label from worker thread safely
        status_label.text = msg

    try:
        status_label.text = "Loading models (this may take a few minutes)..."
        models = await run.io_bound(core.load_all_models, progress)
        STATE["models"] = models
        status_label.text = "Models loaded successfully."
        container.set_visibility(True)
    except Exception as exc:
        status_label.text = f"Failed to load models: {exc}"
        ui.notify(f"Model load error: {exc}", type="negative", multi_line=True)


# ==========================================================
# ANALYSIS HELPERS
# ==========================================================

async def analyze_video(video_path, video_name, results_area):
    if STATE["models"] is None:
        ui.notify("Models are still loading.", type="warning")
        return

    results_area.clear()
    with results_area:
        ui.spinner(size="lg")
        ui.label("Analyzing video...")

    try:
        context, _frames = await run.io_bound(
            core.build_context, video_path, STATE["models"]
        )
    except Exception as exc:
        results_area.clear()
        with results_area:
            ui.label(f"Analysis failed: {exc}").classes("text-red-500")
        return

    if context is None:
        results_area.clear()
        with results_area:
            ui.label("Could not read frames from the video.").classes("text-red-500")
        return

    STATE["context"] = context
    STATE["current_video_name"] = video_name
    render_results(context, video_name, results_area)


def render_results(context, video_name, results_area):
    results_area.clear()
    with results_area:
        ui.label("Analysis Complete").classes("text-xl font-bold text-green-600")

        with ui.row().classes("w-full gap-6 no-wrap"):
            # Classification card
            with ui.card().classes("flex-1"):
                ui.label("Classification").classes("text-lg font-semibold")
                status_color = (
                    "text-red-600" if context["binary_class"] == "ANOMALOUS"
                    else "text-green-600"
                )
                ui.label(
                    f"Status: {context['binary_class']} "
                    f"({context['binary_confidence']:.2%})"
                ).classes(f"font-bold {status_color}")
                ui.separator()
                ui.label(f"People: {context['people']}")
                ui.label(f"Weapon: {context['weapon']}")
                ui.label(f"Location: {context['location']}")
                ui.label(f"Category: {context['category']}")

            # Summary card
            with ui.card().classes("flex-1"):
                ui.label("Summary").classes("text-lg font-semibold")
                ui.label(context["summary"]).classes("text-sm whitespace-pre-wrap")

                def save_summary():
                    base = os.path.splitext(os.path.basename(video_name))[0]
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    txt_path = os.path.join(core.SAVE_DIR, f"{base}_{ts}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(context["summary"])
                    ui.notify(f"Saved to {txt_path}", type="positive")

                ui.button("Save Summary", on_click=save_summary).props("outline")

        # Q&A section
        with ui.card().classes("w-full"):
            ui.label("Ask a Question").classes("text-lg font-semibold")
            qa_log = ui.column().classes("w-full")
            question_input = ui.input("Ask about the video...").classes("w-full")

            def ask():
                q = question_input.value
                if not q:
                    return
                ans = core.answer_question(context, q)
                with qa_log:
                    with ui.card().classes("w-full bg-blue-50"):
                        ui.label(f"Q: {q}").classes("font-semibold")
                        ui.label(f"A: {ans}")
                question_input.value = ""

            question_input.on("keydown.enter", ask)
            ui.button("Ask", on_click=ask)


# ==========================================================
# UI LAYOUT
# ==========================================================

@ui.page("/")
def main_page():
    ui.colors(primary="#1565c0")

    with ui.header().classes("items-center justify-between"):
        ui.label("Video QA & Anomaly Detection").classes("text-xl font-bold")
        ui.label(f"Device: {core.DEVICE}").classes("text-sm opacity-80")

    # Status / loader
    status_label = ui.label("Initializing...").classes("text-sm text-gray-600 p-2")

    # Main content (hidden until models load)
    content = ui.column().classes("w-full p-4 gap-4")
    content.set_visibility(False)

    with content:
        with ui.tabs().classes("w-full") as tabs:
            tab_upload = ui.tab("Upload Video")
            tab_local = ui.tab("Local Video")
            tab_settings = ui.tab("Settings")

        with ui.tab_panels(tabs, value=tab_upload).classes("w-full"):
            # ---- Upload tab ----
            with ui.tab_panel(tab_upload):
                upload_results = ui.column().classes("w-full gap-4")

                async def handle_upload(e):
                    suffix = os.path.splitext(e.name)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(e.content.read())
                        tmp_path = tmp.name
                    await analyze_video(tmp_path, e.name, upload_results)

                ui.upload(
                    label="Choose a video file",
                    on_upload=handle_upload,
                    auto_upload=True,
                ).props('accept="video/*"').classes("w-full")

            # ---- Local video tab ----
            with ui.tab_panel(tab_local):
                local_results = ui.column().classes("w-full gap-4")
                videos = core.list_local_videos()

                if videos:
                    video_select = ui.select(
                        videos, label="Choose a video"
                    ).classes("w-full")

                    async def analyze_local():
                        if not video_select.value:
                            ui.notify("Select a video first.", type="warning")
                            return
                        await analyze_video(
                            video_select.value,
                            os.path.basename(video_select.value),
                            local_results,
                        )

                    ui.button("Analyze Video", on_click=analyze_local)
                else:
                    ui.label(
                        f"No videos found in: {core.VIDEO_DIR}"
                    ).classes("text-orange-600")

            # ---- Settings tab ----
            with ui.tab_panel(tab_settings):
                ui.label("Configuration").classes("text-lg font-semibold")
                ui.code(
                    "\n".join([
                        f"MODEL_DIR   = {core.MODEL_DIR}",
                        f"CHECKPOINT  = {core.CHECKPOINT}",
                        f"DEVICE      = {core.DEVICE}",
                        f"BASE_MODEL  = {core.BASE_MODEL}",
                        f"EMBED_MODEL = {core.EMBED_MODEL}",
                        f"FRAMES      = {core.FRAMES}",
                        f"SAVE_DIR    = {core.SAVE_DIR}",
                        f"VIDEO_DIR   = {core.VIDEO_DIR}",
                    ]),
                    language="text",
                ).classes("w-full")
                ui.label("Required model files:").classes("font-semibold mt-2")
                ui.code("\n".join(core.MODEL_FILES), language="text").classes("w-full")

    # Kick off model loading after the page is ready
    ui.timer(0.1, lambda: load_models_async(status_label, content), once=True)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Video QA System",
        native=True,
        window_size=(1200, 800),
        reload=False,
        port=8521,
    )
