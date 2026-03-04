"""
env/traffic_generator.py

Poisson traffic generator with time-of-day variation.

The arrival rate (lambda) is scaled by a multiplier that depends on
whether the current simulated hour falls within the configured peak
window.  Inter-arrival times are drawn from an Exponential distribution,
which is the waiting-time distribution of a Poisson process.
"""

import numpy as np


class TrafficGenerator:
    """Generates Poisson-distributed request arrivals with time-of-day variation.

    Parameters
    ----------
    base_rate : float
        Base arrival rate in requests-per-second at off-peak hours.
    peak_hours : tuple[int, int]
        (start_hour, end_hour) in 24-h format.  Arrival rate is scaled up
        during this window.
    peak_multiplier : float
        Lambda scaling factor during peak hours.
    offpeak_multiplier : float
        Lambda scaling factor outside peak hours.
    rng : np.random.Generator | None
        Optional seeded RNG for reproducibility.
    """

    def __init__(
        self,
        base_rate: float = 10.0,
        peak_hours: tuple = (8, 18),
        peak_multiplier: float = 2.5,
        offpeak_multiplier: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        self.base_rate = base_rate
        self.peak_hours = peak_hours
        self.peak_multiplier = peak_multiplier
        self.offpeak_multiplier = offpeak_multiplier
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def effective_rate(self, hour_of_day: float) -> float:
        """Return the effective arrival rate (req/s) for the given hour.

        Parameters
        ----------
        hour_of_day : float
            Fractional hour in [0, 24).  For example 14.5 means 14:30.
        """
        start, end = self.peak_hours
        if start <= hour_of_day < end:
            multiplier = self.peak_multiplier
        else:
            multiplier = self.offpeak_multiplier
        return self.base_rate * multiplier

    def next_inter_arrival_ms(self, hour_of_day: float) -> float:
        """Sample the next inter-arrival time (ms) for a Poisson process.

        For a Poisson process with rate λ (req/s), the inter-arrival time
        follows Exponential(λ).  We convert to milliseconds.

        Parameters
        ----------
        hour_of_day : float
            Fractional hour used to pick the effective λ.

        Returns
        -------
        float
            Inter-arrival time in milliseconds.
        """
        rate_per_sec = self.effective_rate(hour_of_day)
        if rate_per_sec <= 0:
            return float("inf")
        rate_per_ms = rate_per_sec / 1000.0
        # Exponential(rate_per_ms): mean = 1/rate_per_ms ms
        return self.rng.exponential(scale=1.0 / rate_per_ms)

    def arrivals_in_window_ms(self, window_ms: float, hour_of_day: float) -> int:
        """Return the number of Poisson arrivals in a fixed time window.

        Uses the analytical Poisson PMF rather than sampling individual
        inter-arrival times; this is useful for bulk steps.

        Parameters
        ----------
        window_ms : float
            Length of the time window in milliseconds.
        hour_of_day : float
            Fractional hour used to determine the effective λ.

        Returns
        -------
        int
            Number of new requests arriving in this window.
        """
        rate_per_sec = self.effective_rate(hour_of_day)
        lambda_window = rate_per_sec * (window_ms / 1000.0)
        return int(self.rng.poisson(lambda_window))
