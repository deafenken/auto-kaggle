# Hero image prompt — `docs/hero.png`

Drop the generated image into `docs/hero.png` (1024×1024 PNG, white-or-very-light
background) and the README will pick it up automatically. The README does not
break without the image — it falls back to the mermaid pipeline.

## Where to send this prompt

- **GPT-image-1** (`openai.images.generate(model="gpt-image-1", size="1024x1024", prompt=…)`)
- **Midjourney** (paste; add `--ar 1:1 --style raw` at the end)
- **Gemini Image** (paste in Gemini Image gen)
- **DALL·E 3** (paste; pick 1024×1024)

## Prompt (paste this verbatim)

> A cartoon-illustration hero banner in the style of a friendly Notion blog
> header, warm Kaggle-teal `#20BEFF` primary palette with gold medal accents
> and small warning-orange highlights, white background, crisp confident
> black linework + watercolor fills, soft drop shadows.
>
> **Left third** — a cartoon hand holding a phone. The phone screen shows a
> Telegram-style chat with one message bubble: `auto kaggle
> playground-series-s4e5`. Behind the phone is a code editor monitor
> displaying snippet of LightGBM Python training code. A small floating
> "📈 leaderboard.csv" sticker.
>
> **Center** — four labeled icons arranged in a clockwise circular flow with
> round connecting arrows: (1) "🗂️ Download + Parse" downloading-cloud icon,
> (2) "📚 Recon Top Kernels" notebook stack with attribution badges,
> (3) "🧪 Train + Ablate" test tubes with CV-fold lines, (4) "🎯 Rank +
> Submit" trophy with a circular "3/5" quota meter.
>
> **Right third** — a small friendly mascot creature (stylized whale or owl,
> not a real animal trademark) wearing a tiny silver medal around its neck,
> holding a stopwatch labeled `00:00 UTC` that points at the daily quota
> meter. Next to the mascot is a curved "wait_until → resume" arrow
> indicating the crash-safe loop.
>
> **Overlay text, large chunky 3D-drop-shadow sans-serif**, two lines:
>
>   *(top)*    **Drop a URL. Hunt medals.**
>   *(bottom small)*    auto-kaggle · multi-day · CV-first · crash-safe
>
> No real logos. No actual Kaggle / OpenAI / Anthropic logos. No
> celebrity faces. Type must remain readable when the image is downscaled
> to 800px on GitHub.

## Optional second image

If you want a "before/after" or "iteration chart" companion image (the
reference had two side by side), the chart is already covered by the mermaid
diagram inside the README. A second image is optional — if you want one, ask
the same tool for a hand-drawn CV-vs-Public-LB scatter plot with a "shake-up"
warning arrow, in the same color palette.
