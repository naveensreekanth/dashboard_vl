"""
LLMEngine (Option A) — Phi-3 Mini 3.8B via llama-cpp-python.
Requires: pip install llama-cpp-python (CUDA build for GTX 1650)
Model file: models/phi3-mini-q4.gguf  (~2.4 GB)
"""

import os
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "phi3-mini-q4.gguf"

PROMPT_TEMPLATE = """<|system|>
You are a semiconductor test engineer writing a professional SHMOO analysis report.
Write clear, technical, precise English. Use flowing paragraphs — no bullet points.
<|end|>
<|user|>
Write the Executive Summary section of a SHMOO analysis report using these results:

- Total test points : {n_total} ({n_pass} PASS, {n_fail} FAIL, {pass_rate:.1f}% pass rate)
- VDD range tested  : {vdd_min:.2f}–{vdd_max:.2f} V
- Freq range tested : {freq_min:.2f}–{freq_max:.2f} GHz
- Boundary (linear) : Fmax(GHz) ≈ {slope:.2f} × VDD(V) {sign}{intercept:.2f}  (R² = {r2:.3f})
- Recommended OP    : VDD = {rec_vdd:.3f} V,  Freq = {rec_freq:.3f} GHz
- Voltage margin    : {v_margin:.0f} mV   |   Frequency margin: {f_margin:.0f} MHz
- Failure breakdown : {fail_breakdown}
- TIMING anomaly    : {timing_info}
- CV accuracy       : {cv_acc:.2f}%

Write 3–4 paragraphs covering:
(1) overall results and boundary characterisation,
(2) failure mode analysis and production implications,
(3) the recommended operating point with margin justification,
(4) caveats / next steps.
<|end|>
<|assistant|>"""


class LLMEngine:
    def __init__(self):
        self._llm = None

    def is_available(self) -> bool:
        return MODEL_PATH.exists()

    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            )
        if not self.is_available():
            raise FileNotFoundError(
                f"LLM model not found at {MODEL_PATH}.\n"
                "Download Phi-3-mini-4k-instruct-q4.gguf from HuggingFace "
                "and place it in the models/ folder."
            )
        self._llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=32,
            n_ctx=2048,
            n_batch=512,
            verbose=False,
        )

    def generate(self, results, meta, test_methodology='MBIST') -> str:
        self._load()

        fail_codes = results.failure_code_dist
        n_fail     = results.n_fail or 1

        fail_breakdown = ', '.join(
            f'{k}: {v} ({v / n_fail * 100:.0f}%)'
            for k, v in fail_codes.items()
            if k != 'NA' or v > 0
        )

        if results.timing_fail_patterns:
            top         = results.timing_fail_patterns[:3]
            pats        = ', '.join(p['pattern'] for p in top)
            timing_info = (
                f"{fail_codes.get('TIMING', 0)} TIMING fails "
                f"concentrated in patterns: {pats}"
            )
        else:
            timing_info = "None detected"

        prompt = PROMPT_TEMPLATE.format(
            n_total=results.n_pass + results.n_fail,
            n_pass=results.n_pass,
            n_fail=results.n_fail,
            pass_rate=results.n_pass / (results.n_pass + results.n_fail) * 100,
            vdd_min=meta['vdd_range'][0],
            vdd_max=meta['vdd_range'][1],
            freq_min=meta['freq_range'][0],
            freq_max=meta['freq_range'][1],
            slope=results.boundary_slope,
            sign='+' if results.boundary_intercept >= 0 else '',
            intercept=results.boundary_intercept,
            r2=results.boundary_r2,
            rec_vdd=results.recommended_vdd,
            rec_freq=results.recommended_freq,
            v_margin=results.voltage_margin_v * 1000,
            f_margin=results.freq_margin_ghz * 1000,
            fail_breakdown=fail_breakdown,
            timing_info=timing_info,
            cv_acc=results.cv_accuracy * 100,
        )

        output = self._llm(
            prompt,
            max_tokens=700,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=['<|end|>', '<|user|>'],
        )
        return output['choices'][0]['text'].strip()
