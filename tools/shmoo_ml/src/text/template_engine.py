"""
TemplateEngine (Option B) — lightweight, no LLM required.
Selects a pre-written narrative template based on result metrics and fills
in the actual numbers. Near-instant, zero extra RAM.
Updated to append Multi-Device performance ranking notes when present.
"""


class TemplateEngine:
    TEMPLATES = {
        'high_r2_good_margin_no_timing': (
            "Across {n_total:,} VDD × Frequency test points "
            "({vdd_min:.2f}–{vdd_max:.2f} V, {freq_min:.2f}–{freq_max:.2f} GHz), "
            "the device passed {n_pass:,} points ({pass_rate:.1f}%) and failed "
            "{n_fail:,} ({fail_rate:.1f}%). The pass/fail boundary is highly linear "
            "(R² = {r2:.3f}): Fmax(GHz) ≈ {slope:.2f} × VDD(V) {sign}{intercept:.2f}. "
            "No significant TIMING anomalies were detected; all failures follow the "
            "expected voltage/frequency-limited pattern.\n\n"
            "The ML model achieved {cv_acc:.1f}% cross-validated accuracy, confirming "
            "reliable boundary detection. Yield rises monotonically with VDD, "
            "indicating a well-characterised, stable device with no anomalous "
            "failure clusters outside the expected voltage-scaling trend.\n\n"
            "Recommendation: Set the production screening operating point at "
            "VDD ≥ {rec_vdd:.3f} V and Frequency ≤ {rec_freq:.3f} GHz, which sits "
            "inside the observed pass region with a voltage margin of {v_margin:.0f} mV "
            "and frequency margin of {f_margin:.0f} MHz. A standard single-tier "
            "frequency/voltage sweep is sufficient for production screening."
        ),

        'high_r2_good_margin_with_timing': (
            "Across {n_total:,} VDD × Frequency test points "
            "({vdd_min:.2f}–{vdd_max:.2f} V, {freq_min:.2f}–{freq_max:.2f} GHz), "
            "the device passed {n_pass:,} points ({pass_rate:.1f}%) and failed "
            "{n_fail:,} ({fail_rate:.1f}%). The pass/fail boundary is highly linear "
            "(R² = {r2:.3f}): Fmax(GHz) ≈ {slope:.2f} × VDD(V) {sign}{intercept:.2f}. "
            "Two distinct failure mechanisms are present — FREQ_MARGIN failures "
            "({freq_margin_count} fails, {freq_margin_pct:.0f}%) track the expected "
            "voltage/frequency boundary, while TIMING failures ({timing_count} fails, "
            "{timing_pct:.0f}%) are concentrated in a pattern-specific subset and "
            "represent a separate risk that a simple frequency guardband will not fully screen.\n\n"
            "The ML model achieved {cv_acc:.1f}% cross-validated accuracy. "
            "TIMING failures are not explained by the linear boundary model and "
            "indicate critical-path sensitivity in specific ATPG patterns, requiring "
            "a dedicated screening step beyond the standard frequency sweep.\n\n"
            "Recommendation: Set the production screening operating point at "
            "VDD ≥ {rec_vdd:.3f} V and Frequency ≤ {rec_freq:.3f} GHz "
            "(voltage margin {v_margin:.0f} mV, frequency margin {f_margin:.0f} MHz). "
            "Implement a two-tier screen: (1) a fast voltage/frequency sweep to catch "
            "FREQ_MARGIN fails, plus (2) a targeted pattern set built from the top "
            "TIMING-failing patterns to catch critical-path fails that frequency "
            "guardband alone would miss."
        ),

        'narrow_margin': (
            "Across {n_total:,} test points, the device passed {n_pass:,} "
            "({pass_rate:.1f}%) and failed {n_fail:,} ({fail_rate:.1f}%). "
            "The boundary is characterised (R² = {r2:.3f}) but operating margins "
            "are narrow — voltage margin {v_margin:.0f} mV, frequency margin "
            "{f_margin:.0f} MHz — indicating limited guardband for production. "
            "The ML model achieved {cv_acc:.1f}% CV accuracy.\n\n"
            "Caution is advised: the narrow margins suggest this device may be "
            "sensitive to process variation. Additional die/wafer characterisation "
            "is strongly recommended before locking production limits. Consider "
            "applying a larger guardband (+30–50 mV VDD, −10% frequency) to ensure "
            "robust yield across process corners."
        ),

        'low_accuracy': (
            "Across {n_total:,} test points, the device passed {n_pass:,} "
            "({pass_rate:.1f}%) and failed {n_fail:,} ({fail_rate:.1f}%). "
            "The ML model achieved {cv_acc:.1f}% cross-validated accuracy — "
            "below the 95% target — suggesting the pass/fail boundary may be "
            "non-linear, noisy, or that additional features are needed for reliable "
            "classification. The boundary fit (R² = {r2:.3f}) should be treated "
            "as an approximation.\n\n"
            "Manual review of the SHMOO plot is recommended. Possible causes include "
            "process corner mixing, temperature variation in the dataset, or "
            "multi-modal failure behaviour. Additional characterisation sweeps are "
            "advised before deriving production screening limits from this data."
        ),
    }

    def _select_template(self, results) -> str:
        cv_ok      = results.cv_accuracy >= 0.95
        has_timing = results.failure_code_dist.get('TIMING', 0) > 0
        margin_ok  = (results.voltage_margin_v  >= 0.05 and
                      results.freq_margin_ghz    >= 0.05)
        if not cv_ok:
            return 'low_accuracy'
        if not margin_ok:
            return 'narrow_margin'
        if has_timing:
            return 'high_r2_good_margin_with_timing'
        return 'high_r2_good_margin_no_timing'

    def generate(self, results, meta, test_methodology='MBIST') -> str:
        key      = self._select_template(results)
        template = self.TEMPLATES[key]

        n_total           = results.n_pass + results.n_fail
        n_fail            = results.n_fail
        fail_codes        = results.failure_code_dist
        freq_margin_count = fail_codes.get('FREQ_MARGIN', 0)
        timing_count      = fail_codes.get('TIMING', 0)

        narrative = template.format(
            n_total=n_total,
            n_pass=results.n_pass,
            n_fail=n_fail,
            pass_rate=results.n_pass / n_total * 100,
            fail_rate=n_fail / n_total * 100,
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
            cv_acc=results.cv_accuracy * 100,
            freq_margin_count=freq_margin_count,
            freq_margin_pct=freq_margin_count / n_fail * 100 if n_fail else 0,
            timing_count=timing_count,
            timing_pct=timing_count / n_fail * 100 if n_fail else 0,
        )

        # Dynamic terminology customization based on selected methodology
        if test_methodology == 'ATPG':
            narrative = (narrative
                .replace("the device passed", f"the device under ATPG Scan passed")
                .replace("anomalous failure clusters", "ATPG scan chain path delay clusters")
                .replace("TIMING failures", "scan path timing delay failures")
                .replace("TIMING-failing patterns", "ATPG critical-path timing patterns")
                .replace("functional/hard defects", "scan timing/path delay defects")
            )
        elif test_methodology == 'LBIST':
            narrative = (narrative
                .replace("the device passed", f"the Logic BIST (LBIST) channel passed")
                .replace("anomalous failure clusters", "LBIST channel transition delay clusters")
                .replace("TIMING failures", "logic gate transition delay failures")
                .replace("TIMING-failing patterns", "LBIST transition timing patterns")
                .replace("functional/hard defects", "gate transition delay defects")
            )
        else: # MBIST
            narrative = (narrative
                .replace("the device passed", f"the Memory BIST (MBIST) array passed")
                .replace("anomalous failure clusters", "MBIST memory array write/read clusters")
                .replace("TIMING failures", "memory functional timing failures")
                .replace("TIMING-failing patterns", "March algorithm timing patterns")
                .replace("functional/hard defects", "memory array/cell defects")
            )

        if getattr(results, 'is_multi_die', False) and getattr(results, 'high_performer', None):
            high = results.high_performer
            low = results.low_performer
            rankings = results.die_rankings
            multi_note = (
                f"\n\nMulti-Device Analysis ({len(rankings)} devices evaluated):\n"
                f"- High Performer: Device '{high['die_id']}' achieved the highest maximum frequency of {high['fmax_at_nom']:.2f} GHz at nominal VDD (yield {high['pass_rate']*100:.1f}%).\n"
                f"- Low Performer: Device '{low['die_id']}' represents the speed-limiting corner with Fmax of {low['fmax_at_nom']:.2f} GHz at nominal VDD (yield {low['pass_rate']*100:.1f}%).\n"
                f"The overall population recommendation (Frequency ≤ {results.recommended_freq:.3f} GHz) is bounded by the Low Performer to guarantee 100% binning yield across all devices."
            )
            narrative += multi_note

        return narrative
