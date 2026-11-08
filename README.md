# UGC Studio

A local, single-machine pipeline that turns a text brief plus product/creator
photos into finished short-form UGC ads with MiniMax H3 in ComfyUI.

Brief → storyboard → scene stills → lip-synced scene videos → stitched export
with burned-in brand, CTA and captions. Every stage has an approval gate; nothing
renders until you approve the previous step.

Nothing leaves your machine. No Python packages to install — standard library
only.

---

## Requirements

| What | Why | Notes |
|---|---|---|
| **Python 3.10+** | runs the app | 3.12/3.13/3.14 all fine. No `pip install` needed. |
| **ComfyUI** | does the generating | Desktop app or portable build. Must be running before you render. |
| **MiniMax H3 models** | the actual model | See [Models](#models). ~30 GB total. |
| **FFmpeg** | stitching + audio | Must be on `PATH` (or set `FFMPEG_BIN`). |
| **A GPU** | 8 GB VRAM minimum | 8 GB runs draft; 16 GB+ makes standard/high comfortable. |

Optional: a Gemini or OpenRouter API key for a smarter storyboard planner. The
built-in offline planner works without one.

---

## Install on a fresh machine

1. **Install Python 3.10+** and tick *"Add python.exe to PATH"* during setup.
2. **Install ComfyUI** and get it launching.
3. **Install the models** into ComfyUI's folders (see [Models](#models)).
4. **Install FFmpeg** and confirm `ffmpeg -version` works in a terminal.
5. **Clone this repo:**

   ```bat
   git clone https://github.com/<you>/ugc-studio.git
   cd ugc-studio
   ```

6. **Create your settings file** (copy the example, then add your keys):

   ```bat
   copy settings.example.json settings.json
   ```

7. **Start ComfyUI**, then double-click **`Launch UGC Studio.bat`**.
   The dashboard opens at <http://127.0.0.1:8756>.

macOS / Linux: `python3 server.py` from the repo folder, then open the same URL.

### Telling the app where ComfyUI lives

The app needs ComfyUI's install folder to find `output/` and
`models/characters/`. It resolves in this order:

1. `UGC_COMFY_DIR` environment variable
2. the `comfy_dir` value in `settings.json`
3. the default `E:\Comfy\ComfyUI\ComfyUI`

So on another machine either set `comfy_dir` in Settings (or `settings.json`), or:

```bat
set UGC_COMFY_DIR=D:\ComfyUI
```

Same idea for FFmpeg: `set FFMPEG_BIN=C:\ffmpeg\bin\ffmpeg.exe` if it is not on
`PATH`.

---

## Models

Everything is optional to download — the app never fetches a model on its own.

| Folder | Files |
|---|---|
| `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`, `pixel_space.safetensors` |
| `models/loras/` | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` |

Open **Model inventory** in the sidebar to see what is installed, what is
missing, and how much VRAM/RAM/disk you have.

> Do **not** install `p-h3-image_lora_bf16` — it corrupts ref2va output.

---

## How a project runs

| Stage | What happens | Gate |
|---|---|---|
| **01 Plan** | The planner reads the brief and writes a shot-by-shot storyboard. Every line is editable. | Approve the storyboard |
| **02 Images** | One still per scene, using your product and creator references. | Approve each still |
| **03 Videos** | Image-to-video with real speech, lip-synced in the same pass. | Review each clip |
| **04 Export** | Scenes normalised to one canvas, stitched, text burned in as real pixels. | Download |

### Tips that save real time

- **Look test first.** On the Images tab, render scene 1 only, check the look,
  approve it, then render the rest. A wrong look costs one render instead of one
  per scene.
- **Turbo off, 20 steps** gives noticeably more photographic skin than turbo on
  at 8–12 steps. It costs roughly 1.7× the time.
- **Draft resolution (384×672) is the right default.** It looks good and is fast;
  upscale the final cut afterwards if you need 1080×1920.
- **Longer videos mean more scenes**, not longer clips — one H3 generation tops
  out near 15 seconds.
- **Spell out brand names in dialogue.** H3 cannot pronounce acronyms like "GTA 6"
  and will read them as gibberish. Write "Grand Theft Auto 6".
- **Keep spoken lines short** — around 12 words. Long clause-heavy lines collapse
  at low step counts.

---

## References

Drop files straight onto the boxes in the Brief tab — no paths to type.

| Box | What it does |
|---|---|
| **Product images** | Locks product identity in every scene. |
| **Creator images** | Steers the face and build. |
| **Scene references** | Locks framing per scene — scene 1 uses image 1, scene 2 image 2. |
| **Extra references** | Anything else that should appear. |

**Import sheet** (Images tab) accepts a character sheet drawn anywhere — Krea,
Blender, a photo — and uses it as the identity anchor for every still.
**Import still** (per scene card) accepts a still made outside the app and puts it
back into the approval flow.

---

## Characters

A `.char` is a reusable actor: face, body and outfit reference images plus an
optional voice clip, encoded into one ~17 MB file in
`models/characters/<name>.char`.

- Create one in the **Characters** tab, then **Use** it in any project.
- It applies to the **video step** of every project, on top of the approved still.
- The still always stays reference slot 0, so the composition and wardrobe you
  approved are never thrown away.

Leave the library cleared if each project has its own creator sheet — that is the
simplest setup and the one most projects want.

---

## Settings and API keys

Settings live in `settings.json` next to `server.py`. It is gitignored, because it
holds your keys in plaintext.

- **Gemini** (primary planner + TTS) — paste the key in Settings.
- **OpenRouter** (fallback planner) — optional.
- **No key at all** — the offline rule-based planner still writes a usable
  storyboard.

Cloud planning sends **brief text only**. Images are never uploaded.

---

## Where your work lives

```
projects/<project-id>/
  project.json          all state: brief, storyboard, approvals, settings used
  brief/                brief.txt, brief_fields.json
  references/           products/ creators/ scenes/ extra/ videos/
  storyboard/           storyboard.json, storyboard.md
  scene_images/scene_01/approved.png
  scene_videos/scene_01/scene_01_<stamp>.mp4
  audio/scene_01.wav
  prompts/              the exact graph inputs used per scene
  exports/              final stitched cuts
  logs/studio.log
```

`projects/` and `uploads/` are gitignored — they are your work, not the app.

---

## Troubleshooting

**"ComfyUI is not running"** — start ComfyUI first, then refresh. Check
`comfy_url` in Settings if it is not on `127.0.0.1:8188`.

**"Missing nodes"** — the ComfyUI custom nodes listed in Model inventory are not
installed. Install them in ComfyUI Manager and restart ComfyUI.

**Video step fails on a long scene** — one generation caps near 15 seconds. Split
the scene, or use **One continuous take** only for short ads.

**Port 8756 is busy** — `set UGC_PORT=8757` before launching.

**Planner returns nothing** — no key configured and the brief is too vague for the
offline planner. Add a key, or write a clearer brief.

**Windows settings got wiped** — never write `settings.json` with PowerShell's
`Set-Content -Encoding utf8`; the BOM breaks JSON parsing. Edit it in the app, or
save it as UTF-8 without BOM.

---

## Licence

Private project. MiniMax H3, ComfyUI and FFmpeg carry their own licences — check
them before commercial use. If you use a real person's face or voice, get written
consent, and disclose realistic AI-generated humans on Meta/TikTok as their ad
policies require.
