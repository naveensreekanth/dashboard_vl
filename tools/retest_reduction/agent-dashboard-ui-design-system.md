# Professional Agent Dashboard UI Design System — Prompt Template

A reusable design-system prompt for making internal AI-agent dashboards look like professional, human-designed engineering tools instead of generic "AI SaaS" templates. Paste the relevant sections into your coding agent (along with a screenshot of your current UI) and fill in the bracketed placeholders for your own product.

---

## 1. Design principles

- **Calm, not flashy.** The tool should look like something a domain expert (engineer, analyst, operator) would trust in production — dense, high-contrast where it matters, restrained color. Not a consumer app, not a hackathon demo.
- **One accent color, used sparingly.** Pick a single accent for primary actions and active/selected states. Don't repeat it on every card, badge, and button — that's the #1 tell of a generated/templated UI.
- **Semantic color, not decorative color.** Color should communicate meaning (pass/fail, good/warning/critical, active/inactive), not just decorate. If a color doesn't mean something, it shouldn't be there.
- **Borders over glow.** Hairline 1px low-opacity borders read as engineering tooling. Glowing box-shadows and heavy gradients read as AI-generated.
- **Panels and lists over card-per-metric.** Don't wrap every single number in its own bordered card. Use a small number of large cards for true top-line numbers, and grouped stat-list rows (label left, value right) or breakdown visuals for everything else.

## 2. Color system

Pick **one** base theme and apply it consistently everywhere — don't mix a light sidebar with a dark content area or vice versa.

**Dark / engineering-terminal theme**
- Base background: near-black slate/graphite, e.g. `#0B0D12`
- Panel/card background: one or two steps lighter, e.g. `#12162A`
- Borders: 1px, low-opacity white/gray, e.g. `#233044`
- Accent: one restrained color for primary actions + active nav state (industrial blue or teal reads more "engineering" than purple)

**Light / clean-enterprise theme**
- Base background: soft gray-white, e.g. `#F4F6FA`
- Panel/card background: white, with thin low-opacity gray borders, no shadow glow
- Sidebar: dark navy, with a single accent color for the active nav item
- Accent: one confident color (teal, blue) for primary actions and highlights only

**Semantic colors (apply consistently regardless of theme):**
- Green — good outcome / recommended / pass
- Amber/orange — warning / needs attention / inefficiency
- Red — critical / failure / do-not-proceed
- Neutral gray-blue — informational, non-actionable stats

## 3. Typography

- Use a real type scale: page title → section header → panel label → value → caption, each a clear step down in size/weight — not everything the same medium weight.
- **Numeric/data values should use a monospace or tabular-nums font** (IBM Plex Mono, JetBrains Mono, or Inter/system font with tabular-nums enabled) so numbers align in columns. This single change does more than almost anything else to make a data tool look professional instead of generic.
- Section/group labels (sidebar section headers, panel eyebrow labels) should be small, uppercase, letter-spaced, and lower-contrast (muted gray) — they should support the data, not compete with it.

## 4. Layout & composition patterns

- **Top-line cards, sparingly.** Reserve bordered/boxed cards for the 3–5 numbers that matter most at a glance (e.g. total items processed, primary recommendation count, estimated impact/savings).
- **Grouped stat-list rows for everything else.** Put related secondary metrics inside a single panel as label-left/value-right rows with dividers, rather than each getting its own card.
- **One large breakdown visual instead of four small cards.** For any 2x2 or confusion-matrix-style result (e.g. recommendation vs. actual outcome), render it as a single large color-coded grid/visual rather than four separate stat cards — it communicates the relationship between the numbers, not just the numbers themselves.
- **Horizontal bar lists for categorical breakdowns** (counts by category/type) instead of a bar chart component with heavy chrome — a simple aligned label + bar + value list is cleaner and more scannable.
- **Section grouping and dividers.** On a long single-page dashboard, add clear visual separation between distinct sections (overview / breakdown / validation / settings) so it doesn't read as one undifferentiated scroll. A sticky mini-nav or "jump to section" affordance helps on data-dense pages.

## 5. Components

- **Sidebar navigation:** dark base, section headers grouped with clear separation, active item indicated with a left-border accent + subtle background tint (not a full solid-color pill). Slim, no icon glow.
- **Buttons:** one solid-filled style reserved for true primary actions (e.g. "Run analysis," "Upload"). Everything else — secondary actions like "View details" or "Inspect" — should be outline/ghost/text-style buttons. Repeating a solid full-width CTA on every card is a strong "templated" signal.
- **Workflow/progress tracker:** if the tool guides a user through a multi-step process, render it as a real progress tracker (numbered steps, done/active/pending states, one-line descriptions) rather than a plain checklist — and place the relevant action button next to the step it belongs to, not off in an unrelated column.
- **Upload / file-input zones:** give these a visually distinct treatment (e.g. dashed border, drop-zone styling) so they're immediately distinguishable from display-only panels.
- **Status indicators:** a small solid dot + label (not a full colored pill) is the standard convention for system/model status in engineering tools.
- **Tooltips / info icons:** add a small (ⓘ) next to any metric whose meaning isn't self-evident, so the tool is approachable to new users without cluttering the layout for experts.

## 6. Usability checklist

- [ ] Can a first-time user tell at a glance what stage of the workflow they're in?
- [ ] Are related numbers grouped so a user can answer their actual question (e.g. "should I act on this?") without hunting across the page?
- [ ] Is there a clear visual difference between "input" areas (uploads, forms) and "output" areas (results, stats)?
- [ ] Does the active/selected state in navigation stand out clearly from hover states?
- [ ] Is there a way to jump to a section on long pages instead of only scrolling?
- [ ] Do secondary actions look visually secondary (not the same solid CTA style as primary actions)?

## 7. Do's and don'ts

**Don't:**
- Use the same accent color on every card, badge, nav item, and button
- Use heavy glow/box-shadow on buttons or cards
- Wrap every single metric in its own bordered card
- Mix icon styles or stroke weights
- Use arbitrary spacing values — pick a 4px/8px grid and stick to it

**Do:**
- Reserve color for meaning, not decoration
- Use hairline borders to separate panels
- Use tabular/monospace numerals for data
- Group secondary stats into shared list panels
- Keep one consistent icon set at one consistent weight

## 8. How to use this prompt

1. Attach a screenshot of your current dashboard.
2. State your scope explicitly: **visual/styling-only** (restyle existing components, don't touch data/logic) vs. **full rebuild** (new frontend framework/structure, same underlying data).
3. If you have a design reference (another tool's screenshot), say what to *take* from it (theme, color use, specific patterns) and what *not* to copy literally (e.g. "don't copy its exact card layout, just its restraint with color").
4. Name your target aesthetic with 2–3 concrete comparisons (e.g. "like Cadence / Keysight PathWave / a Bloomberg-terminal-style internal tool") — comparisons ground the request far better than adjectives like "modern" or "clean" alone.
5. List what must stay unchanged (all current metrics/features present and functional) so the agent doesn't drop functionality while restyling.

---

### Fill-in-the-blank starter prompt

```
Redesign the UI for [PRODUCT NAME], an internal dashboard for [WHAT IT DOES / WHO USES IT].

Scope: [visual-only restyle / full frontend rebuild in FRAMEWORK].

Problem with current UI: [describe what's generic/templated about it — e.g. one repeated
accent color everywhere, glow effects, everything in identical cards].

Target aesthetic: professional [DOMAIN] tooling, comparable to [2-3 reference tools/products].
Dense, data-serious, calm, restrained color use, human-designed feel — not a generic AI SaaS
template.

Apply the design system in this document: color system, typography, layout & composition
patterns, components, and usability checklist.

Keep unchanged: [list of features/metrics/data that must remain present and functional].

Attached: [screenshot(s) of current UI] [optional: design reference screenshot, noting what
to take from it and what not to copy literally].
```
