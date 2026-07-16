## Changes Made So Far

### Apex-Duration Labeling Workflow

Apex and endpoint labeling now uses `A`:

    A -> select or edit apex
    A -> select or edit endpoint
    A -> apex and endpoint complete


- Apex can be clicked repeatedly until correct.
- Endpoint can be clicked repeatedly until correct.
- Clicking does not advance stages.
- Pressing `A` explicitly advances the stage.
- Endpoint time comes from the user click.
- Duration is calculated from apex and endpoint time.

### Min Max Distance Labeling Workflow

Distance labeling now uses `D`:

    D -> select or edit first distance point
    D -> select or edit second distance point
    D -> distance labeling complete


- Distance point order does not matter.
- Distance minimum and maximum are calculated automatically.
- Clicking does not advance stages.
- Pressing `D` explicitly advances the stage.
- Editing the first distance point clears the second point.

### Text Display Changes

The text display panel now includes an annotation prompt below Cursor Mode. It shows:

- Current annotation stage.
- Next required action.
- Validation errors.
- Selected apex, endpoint, and distance values.
- Completion messages.

### Temporary T-X Plot Markers

The T-X plot now uses:

| Marker | Meaning |
|---|---|
| Red star | Apex |
| Cyan X | Endpoint |
| Yellow circles | Distance boundary points |

Temporary markers are cleared when starting a new annotation, saving an annotation, or pressing `Esc`.

## Not Yet Implemented

The following feature is intentionally excluded from the current workflow:

    Manual F-X bounding-box labeling across multiple F-X time slices,
    with global f_min and f_max calculated from all boxes.