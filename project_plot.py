"""Select an installed CJK font when regenerating Chinese research plots."""
from pathlib import Path

from matplotlib import font_manager, rcParams


def configure_chinese_font():
    windows = Path('C:/Windows/Fonts/msyh.ttc')
    if windows.exists():
        font_manager.fontManager.addfont(str(windows))
        family = font_manager.FontProperties(fname=str(windows)).get_name()
    else:
        family = None
        for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Source Han Sans SC', 'PingFang SC', 'Arial Unicode MS'):
            try:
                font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
                family = name
                break
            except ValueError:
                continue
    if family:
        rcParams['font.family'] = family
    rcParams['axes.unicode_minus'] = False
    return family
