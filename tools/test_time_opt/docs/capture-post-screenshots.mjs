import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = path.join(__dirname, "demo-post-process.html");
const outDir = path.join(__dirname, "images");
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto("file://" + html.replace(/\\/g, "/"), { waitUntil: "networkidle" });

const shots = await page.$$("[data-shot]");
for (const el of shots) {
  const name = await el.getAttribute("data-shot");
  const file = path.join(outDir, `${name}.png`);
  await el.scrollIntoViewIfNeeded();
  await page.waitForTimeout(100);
  await el.screenshot({ path: file });
  console.log("saved", name);
}

await browser.close();
