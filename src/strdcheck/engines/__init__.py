"""Engine adapters (milestones M3-M5).

Each adapter maps a parsed StRD dataset to one engine call path (for example
``polars.lazy.streaming`` or ``scipy.least_squares.trf.analytic.tight``) and returns the
quantities NIST certifies, plus the engine's own status flags. Adapters import their engine
lazily so the core package depends on numpy only. Specification: ``docs/DESIGN.md``
section 5.
"""
