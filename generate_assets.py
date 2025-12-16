import base64
import os

# Source paths (artifacts)
sources = {
    "hollywood": "/Users/damiianp/.gemini/antigravity/brain/f0dd2379-2095-49fc-bf2f-d1c196e0a55b/ref_hollywood_1765904036608.png",
    "natural": "/Users/damiianp/.gemini/antigravity/brain/f0dd2379-2095-49fc-bf2f-d1c196e0a55b/ref_natural_1765904078598.png",
    "alignment": "/Users/damiianp/.gemini/antigravity/brain/f0dd2379-2095-49fc-bf2f-d1c196e0a55b/ref_alignment_1765904120319.png"
}

output_file = "hf_deploy/embedded_assets.py"

content = []
content.append("# Auto-generated embedded assets")
content.append("# These are Base64 encoded images to bypass git-lfs and download issues")
content.append("")
content.append("ASSETS = {")

for key, path in sources.items():
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
            content.append(f'    "{key}": "{b64_string}",')
    else:
        print(f"ERROR: Missing source file {path}")

content.append("}")
content.append("")

with open(output_file, "w") as f:
    f.write("\n".join(content))

print(f"Successfully generated {output_file}")
