"""Adds captions to the rendered frames and encodes tutorial/velvet_collar_tutorial.mp4.

    python3 assemble_video.py            (needs Pillow and ffmpeg)
"""
import glob
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "tutorial", "frames")
COMP = os.path.join(HERE, "tutorial", "captioned")
OUT = os.path.join(HERE, "tutorial", "velvet_collar_tutorial.mp4")
W, H, FPS_SRC = 1280, 720, 15
NAVY, CORAL, TEAL = (22, 34, 64), (240, 98, 92), (64, 190, 170)
SHIFT = 64

# shot prefix, step label, title, instruction, seconds to hold the first frame, seconds to hold the last frame
SHOTS = [
    ("01_supplies", "SUPPLIES", "What you need",
     "1 in (2.5 cm) webbing, 7/8 in (2.2 cm) velvet ribbon, a slider, a side-release buckle, a welded D-ring and "
     "strong polyester thread. Every piece of hardware matches the 1 in webbing.", 0, 6.5),
    ("02_cut", "STEP 1", "Cut the webbing and seal the ends",
     "Cut 26 in (66 cm) for a 16-20 in (40.6-50.8 cm) collar. Use this as a starting point and confirm it with one "
     "test collar. Heat-seal the cut ends: synthetic webbing only, ventilated, water nearby.", 0, 7.5),
    ("03_ribbon", "STEP 2", "Centre the velvet ribbon",
     "Lay the ribbon on the decorative face of the webbing. Keep an even webbing border on both sides.", 1.5, 3.5),
    ("04_edge", "STEP 3", "Edge-stitch the ribbon",
     "Stitch along one long edge, across the end, back along the other edge and across the first end. "
     "Match the bobbin thread to the webbing.", 1.5, 3.5),
    ("05_flip", "STEP 4", "Turn the strap over",
     "The next steps are worked with the velvet side facing down.", 1.2, 2.5),
    ("06_slider", "STEP 5", "Loop one end round the slider",
     "Pass the end under the slider, up through the far opening, over the centre bar and down through the near "
     "opening. Leave about 2 in (5.1 cm) of tail.", 2.0, 4.5),
    ("07_boxx", "STEP 6", "Sew the tail down",
     "Lay the tail flat on the strap and sew a square with an X. Keep the presser foot and needle clear of the slider.",
     1.5, 4.0),
    ("08_male", "STEP 7", "Add the male buckle half",
     "Take the free end up through the slot of the male buckle half and fold it back over the bar.", 1.5, 3.0),
    ("09_fold", "STEP 8", "Feed the strap round the buckle bar",
     "Slide the buckle towards the slider. The returning strap now has its velvet side facing up.", 1.0, 2.5),
    ("10_thread", "STEP 9", "Return through the slider",
     "Lift the slider. Pass the free end under the first bar, over the centre bar on top of the sewn loop, and under "
     "the last bar. Velvet out, no twists.", 2.0, 5.0),
    ("11_pull", "STEP 10", "Pull the strap through",
     "Pull until both layers lie flat. Moving the slider along the strap now changes the collar length.", 1.2, 3.5),
    ("12_dring", "STEP 11", "Slide on the D-ring first",
     "Slide the welded D-ring onto the free end, keeping the strap flat and the velvet facing up.", 1.5, 3.0),
    ("13_female", "STEP 12", "Add the female buckle half",
     "Take the end over the bar and down through the slot. Pull about 4 in (10.2 cm) through and lay the tail back "
     "underneath, under the D-ring bar, so the cut end is hidden on the inside.", 2.5, 5.0),
    ("14_sew", "STEP 13", "Reinforce the fixed end",
     "Sew one row between buckle and D-ring, two rows behind the D-ring, and a square with an X at the tail end. "
     "No stitch may cross a hardware edge.", 2.0, 4.5),
    ("15_flat", "CHECK", "Inspect before use",
     "Velvet faces out with no twists. The D-ring is trapped and cannot travel. No raw ends, skipped stitches or "
     "bunching. The slider changes the length and then holds.", 0, 7.0),
    ("16_final", "FINISHED", "Close the buckle and fit",
     "Fit with two-finger clearance and supervise first use. Retire the collar if the buckle cracks, the slider "
     "slips, the D-ring deforms, the webbing frays or stitching loosens.", 1.0, 6.0),
]


def font(weight, size):
    for p in (f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight != "Regular"
              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_STEP, F_TITLE, F_BODY, F_BRAND = font("Bold", 19), font("Bold", 33), font("Regular", 22), font("SemiBold", 18)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur); cur = word
    return lines + [cur]


def caption(img, step, title, body):
    src = img.convert("RGB").resize((W, H))
    img = Image.new("RGB", (W, H), NAVY)
    img.paste(src, (0, -SHIFT))                      # lift the picture so the caption bar does not cover the subject
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    lines = wrap(d, body, F_BODY, W - 96)
    bar_h = 78 + 30 * len(lines) + 22
    y0 = H - bar_h
    d.rectangle((0, y0, W, H), fill=NAVY + (224,))
    d.rectangle((0, y0, W, y0 + 4), fill=TEAL + (255,))
    pw = int(d.textlength(step, font=F_STEP)) + 28
    d.rounded_rectangle((48, y0 + 24, 48 + pw, y0 + 58), radius=17, fill=CORAL + (255,))
    d.text((62, y0 + 29), step, font=F_STEP, fill="white")
    d.text((48 + pw + 16, y0 + 19), title, font=F_TITLE, fill="white")
    for i, ln in enumerate(lines):
        d.text((48, y0 + 74 + 30 * i), ln, font=F_BODY, fill=(226, 232, 240, 255))
    d.rounded_rectangle((28, 22, 28 + int(d.textlength("Collarly  |  Velvet Ribbon Collar", font=F_BRAND)) + 28, 58),
                        radius=18, fill=(255, 255, 255, 215))
    d.text((42, 28), "Collarly  |  Velvet Ribbon Collar", font=F_BRAND, fill=NAVY + (255,))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


def title_card():
    hero = Image.open(os.path.join(HERE, "renders", "hero.png")).convert("RGB")
    w, h = hero.size; ch = int(w * 9 / 16); top = (h - ch) // 2 + 40
    img = hero.crop((0, top, w, top + ch)).resize((W, H))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    d.rectangle((0, 0, W, 150), fill=NAVY + (228,)); d.rectangle((0, 150, W, 154), fill=TEAL + (255,))
    d.text((48, 26), "Velvet Ribbon Dog Collar", font=font("Bold", 54), fill="white")
    d.text((50, 96), "Step-by-step build: webbing core + velvet ribbon + slider + buckle + D-ring",
           font=font("Regular", 25), fill=(255, 190, 186, 255))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


def main():
    os.makedirs(COMP, exist_ok=True)
    for f in glob.glob(os.path.join(COMP, "*.jpg")):
        os.remove(f)
    entries = []                                     # (path, seconds)
    p = os.path.join(COMP, "00_title.jpg"); title_card().save(p, quality=94); entries.append((p, 3.5))
    total = 3.5
    for prefix, step, title, body, hold0, hold1 in SHOTS:
        frames = sorted(glob.glob(os.path.join(FRAMES, prefix + "_*.jpg")))
        if not frames:
            print("missing shot:", prefix); continue
        for i, f in enumerate(frames):
            p = os.path.join(COMP, os.path.basename(f))
            caption(Image.open(f), step, title, body).save(p, quality=93)
            dur = 1.0 / FPS_SRC + (hold0 if i == 0 else 0) + (hold1 if i == len(frames) - 1 else 0)
            entries.append((p, dur)); total += dur
    lst = os.path.join(COMP, "list.txt")
    with open(lst, "w") as fh:
        for p, dur in entries:
            fh.write(f"file '{p}'\nduration {dur:.4f}\n")
        fh.write(f"file '{entries[-1][0]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "19", "-preset", "medium",
                    "-movflags", "+faststart", OUT], check=True)
    print(f"wrote {OUT}  ({total:.0f} s, {len(entries)} images)")


if __name__ == "__main__":
    main()
