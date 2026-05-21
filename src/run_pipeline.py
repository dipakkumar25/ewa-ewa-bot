import subprocess
import sys
import os
from datetime import datetime


def log(msg):
    """Logger with timestamp"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_command(command, step_name):
    """Run subprocess safely"""
    try:
        log(f"Starting: {step_name}")
        subprocess.run(
            command,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        log(f"Completed: {step_name} ✅")
    except subprocess.CalledProcessError as e:
        log(f"Error in {step_name}: {e}")
        sys.exit(1)


def run_pipeline():
    """Run full EWA pipeline"""

    log("===== EWA Intelligent Hub Pipeline Started =====")

    # Step 1 — HTML Processor
    run_command(
        ["python", "-m", "src.ewa_html_processor"],
        "HTML Processor"
    )

    # Step 2 — KPI Builder
    run_command(
        ["python", "-m", "src.ewa_kpi_master_builder"],
        "KPI Master Builder"
    )

    log("===== Pipeline completed successfully ✅ =====")


def start_streamlit():
    """Start Streamlit UI"""
    log("🌐 Starting Streamlit App...")

    os.execvp(
        "streamlit",
        [
            "streamlit",
            "run",
            "src/ewa_compare_app_enhanced.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ],
    )


if __name__ == "__main__":

    OUTPUT_FILE = "data/ewa_kpi_clean_summary_all.csv"

    # Optional: Skip pipeline if already executed
    if not os.path.exists(OUTPUT_FILE):
        run_pipeline()
    else:
        log("✅ Pipeline already executed. Skipping...")

    # Always start Streamlit
    start_streamlit()