"""Standard HU window / level presets for medical CT imaging.

Each entry: preset_name → (window_center_HU, window_width_HU)
"""

WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    # Neurological
    "Cerebro":       (40.0,   80.0),   # Brain parenchyma
    "Hemorragia":    (55.0,  100.0),   # Intracranial haemorrhage
    "Subdural":      (75.0,  215.0),   # Subdural haematoma
    "CTA":          (170.0,  600.0),   # CT Angiography — vessels
    # Skull / bone
    "Hueso":        (400.0, 1000.0),   # Cortical bone
    # Thorax
    "Pulmón":      (-600.0, 1500.0),   # Lung parenchyma
    "Mediastino":   (50.0,  350.0),    # Mediastinum / soft tissue
    # Abdomen
    "Abdomen":      (40.0,  350.0),    # General soft tissue
    "Hígado":       (70.0,  170.0),    # Liver
}

DEFAULT_PRESET = "Cerebro"
