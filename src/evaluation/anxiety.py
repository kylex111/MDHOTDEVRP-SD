"""Versioned paper RA formulas and normalization; tau is in minutes."""

from __future__ import annotations
import math

RAW_PAPER_RA_VERSION = "paper_exp_v1"
PAPER_RA_VERSION = "paper_exp_norm_v1"


class RangeAnxietyModel:
    def __init__(
        self,
        soc_max: float,
        phi: float = 1.0,
        theta_k: float = 2.0,
        beta_T: float = 1.0,
        kappa_peak: float = 1.2,
        kappa_offpeak: float = 1.0,
        peak_hours=(7, 9, 17, 19),
        *,
        tau: float | None = None,
        formula_version: str = PAPER_RA_VERSION,
    ):
        if formula_version not in (RAW_PAPER_RA_VERSION, PAPER_RA_VERSION):
            raise ValueError("Unknown paper RA formula version")
        if tau is None:
            raise ValueError(
                "Set tau (minutes) explicitly for the paper RA formula; old T0 is not tau"
            )
        values = dict(
            soc_max=soc_max,
            phi=phi,
            theta_k=theta_k,
            beta_T=beta_T,
            kappa_peak=kappa_peak,
            kappa_offpeak=kappa_offpeak,
            tau=tau,
        )
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if soc_max <= 0 or tau <= 0 or phi <= 0 or (theta_k <= 0) or (beta_T < 0):
            raise ValueError("Require B, tau, phi, theta > 0 and beta_T >= 0")
        if not 0 < kappa_offpeak <= kappa_peak:
            raise ValueError("Require kappa_peak >= kappa_offpeak > 0")
        if len(peak_hours) != 4 or any((not math.isfinite(h) for h in peak_hours)):
            raise ValueError("peak_hours must contain four finite hour boundaries")
        sm, em, se, ee = peak_hours
        if not 0 <= sm < em <= se < ee <= 24:
            raise ValueError(
                "Require 0 <= morning_start < morning_end <= evening_start < evening_end <= 24"
            )
        self.soc_max, self.phi, self.theta_k = (
            float(soc_max),
            float(phi),
            float(theta_k),
        )
        self.beta_T, self.tau = (float(beta_T), float(tau))
        self.kappa_peak, self.kappa_offpeak = (float(kappa_peak), float(kappa_offpeak))
        self.peak_hours = tuple(peak_hours)
        self.formula_version = formula_version

    def _energy_fraction(self, energy):
        tolerance = max(1e-09, self.soc_max * 1e-09)
        if (
            not math.isfinite(energy)
            or not -tolerance <= energy <= self.soc_max + tolerance
        ):
            raise ValueError("Consumed energy must lie in [0, battery capacity]")
        return min(1.0, max(0.0, energy / self.soc_max))

    def _gQ(self, Qtr):
        return self.phi * self._energy_fraction(Qtr) ** self.theta_k

    def _gT(self, Ttr):
        if not math.isfinite(Ttr) or Ttr < 0:
            raise ValueError("Travel time must be finite and nonnegative (minutes)")
        return 1.0 - self.beta_T * math.expm1(-Ttr / self.tau)

    def _gh(self, hour_of_day):
        if hour_of_day is None:
            return self.kappa_offpeak
        if not math.isfinite(hour_of_day):
            raise ValueError("Travel hour must be finite")
        hour = hour_of_day % 24.0
        sm, em, se, ee = self.peak_hours
        return (
            self.kappa_peak
            if sm <= hour < em or se <= hour < ee
            else self.kappa_offpeak
        )

    def R_instant(self, Qtr, Ttr, hour_of_day):
        """Instantaneous RA; normalized version lies in [0, 1]."""
        raw = self._gQ(Qtr) * self._gT(Ttr) * self._gh(hour_of_day)
        if self.formula_version == RAW_PAPER_RA_VERSION:
            return raw
        return raw / self._instant_max()

    def _instant_max(self):
        return self.phi * (1.0 + self.beta_T) * self.kappa_peak

    def SUMR(self, Qend, Ttr, hour_of_day):
        """Accumulated RA at consumed energy Qend, with T and h fixed.

        The raw version integrates raw psi over kWh. The normalized version
        integrates normalized psi over consumed battery fraction E/B.
        A segment contributes SUMR(E_after, T, h) - SUMR(E_before, T, h).
        """
        x = self._energy_fraction(Qend)
        raw = (
            self.phi
            * self._gT(Ttr)
            * self._gh(hour_of_day)
            * self.soc_max
            * x ** (self.theta_k + 1.0)
            / (self.theta_k + 1.0)
        )
        if self.formula_version == RAW_PAPER_RA_VERSION:
            return raw
        return raw / (self.soc_max * self._instant_max())
