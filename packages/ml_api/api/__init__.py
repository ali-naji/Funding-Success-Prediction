from api.config import ROOT

with open(ROOT / 'VERSION') as f:
    __version__ = f.read().strip()
