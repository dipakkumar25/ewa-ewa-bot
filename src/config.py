import os

# Detect the project root folder (ewa-ewa-bot)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder paths
HTML_DIR = os.path.join(BASE_DIR, "data", "html")
DETAIL_CSV = os.path.join(BASE_DIR, "data", "ewa_html_traffic_lights_A1C.csv")
SUMMARY_CSV = os.path.join(BASE_DIR, "data", "ewa_html_traffic_lights_summary13_A1C.csv")

# Ensure paths exist
os.makedirs(os.path.dirname(DETAIL_CSV), exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)

print("✔ Config loaded successfully!")
print("📂 HTML_DIR:", HTML_DIR)
print("📄 DETAIL_CSV:", DETAIL_CSV)
print("🗂 SUMMARY_CSV:", SUMMARY_CSV)
