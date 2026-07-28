# Editable Visio paper figures

These scripts generate native, editable Visio shapes through COM automation.
They do not embed PNG, SVG, or other external artwork in the VSDX files.

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\visio\build_fig1_online_interaction.ps1
powershell -ExecutionPolicy Bypass -File tools\visio\build_fig2_environment_model.ps1
```

Outputs are written outside the repository to:

```text
C:\Users\Dk\Desktop\SCI\paper_picture\visio_outputs
```

Figure 1 is the online decision and environment-interaction loop only. Figure 2
is a fixed 15-by-15 local occupancy-grid scene with a one-cell robot, the full
eight-neighbor candidate action space, and 32 representative equal-length rays
for schematic visualization. The ray count is a drawing choice, not a physical
sensor-channel specification.

Each script exports a 300 DPI PNG preview from Visio and then reopens the VSDX
in an independent hidden Visio process for structural validation. If a target
file is open or otherwise locked, the script leaves it unchanged and writes a
timestamped VSDX/PNG pair.
