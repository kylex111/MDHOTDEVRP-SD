"""Operating risk diagnostics; no additional objective or search budget."""

def low_soc_duration(start_kwh, end_kwh, threshold_kwh, duration_min):
    """Integrate time below a threshold within a constant-rate energy segment."""
    if duration_min < 0:
        raise ValueError("Negative segment duration")
    if start_kwh == end_kwh:
        return float(duration_min if start_kwh < threshold_kwh else 0.0)
    if max(start_kwh, end_kwh) <= threshold_kwh:
        return float(duration_min)
    if min(start_kwh, end_kwh) >= threshold_kwh:
        return 0.0
    fraction = (threshold_kwh - min(start_kwh, end_kwh)) / abs(end_kwh - start_kwh)
    return float(duration_min * fraction)
