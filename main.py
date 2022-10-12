from analysis.wayfinding import Wayfinding
from pathlib import Path
W = Wayfinding()
W.import_everything(Path("."))
W.analyze_all_subject()