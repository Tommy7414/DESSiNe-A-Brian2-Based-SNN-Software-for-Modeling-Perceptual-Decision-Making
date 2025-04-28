# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('data', '.')]
hiddenimports = []
datas += collect_data_files('brian2')
datas += collect_data_files('Cython')
datas += collect_data_files('pandas')
hiddenimports += collect_submodules('Cython')


a = Analysis(
    ['AdvancedPlot_GUI.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PDMSNN_ELApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['PDMSNN_EL.icns'],
)
app = BUNDLE(
    exe,
    name='PDMSNN_ELApp.app',
    icon='PDMSNN_EL.icns',
    bundle_identifier=None,
)
