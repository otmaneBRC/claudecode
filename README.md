# Stitchproof landing page

Single-file, mobile-first landing page (`index.html`) — no build step. Open it in a browser or deploy the folder to any static host (Netlify, Vercel, Cloudflare Pages, GitHub Pages).

## Before you launch

1. **Checkout link** — set `CHECKOUT_URL` in the `<script>` at the bottom of `index.html`. Until it's set, CTAs scroll to the offer section.
2. **Localize + compress images** — the page currently hotlinks the 14 Higgsfield-generated photos (4–5 MB PNGs each). Run:
   ```
   pip install pillow
   python3 tools/localize_images.py
   ```
   This downloads them, converts to ~1600px WebP, saves to `assets/img/`, and rewrites `index.html`.
3. Replace the `Terms` / `Privacy` / `Support` footer links and the `og:image` meta tag (point it at a local image after step 2).

## Notes

- Brand tokens (dark brown / tan / cream / black / gold) are CSS variables in `:root`.
- Fonts: Cormorant Garamond (display) + Inter (body) via Google Fonts, with system fallbacks.
- The "Project Blueprint" in section 6 is an illustrative sample sheet built in HTML/CSS (labelled *Example*), not real product specs.
- Image prompts contain no text/logos on purpose; all typography is live HTML.
