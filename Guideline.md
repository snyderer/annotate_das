## Changes Made So Far

### Apex-Duration Labeling Workflow

Apex and endpoint labeling uses `A`:

    A -> select or edit apex
    A -> select or edit endpoint
    A -> apex and endpoint complete

- Apex can be clicked repeatedly until correct.
- Endpoint can be clicked repeatedly until correct.
- Clicking does not advance stages.
- Pressing `A` explicitly advances the stage.
- Endpoint time comes from the user click.
- Endpoint is snapped to the apex channel when clicked near that channel.
- Duration is calculated from apex and endpoint time.

### Min-Max Distance Labeling Workflow

Distance labeling uses `D`:

    D -> enter distance-point selection mode
    Click -> select first distance point
    Click -> select second distance point
    Additional clicks -> keep only the latest two distance points
    D -> distance labeling complete

- Press `D` after apex and endpoint labeling is complete to start distance labeling.
- Click two points on the lower side of the whale call in the T-X plot.
- The two most recently clicked points are retained.
- Distance point order does not matter.
- Distance minimum and maximum are calculated automatically from the two retained points.
- Press `D` again to confirm that the currently retained two points define the distance range.
- The two selected points are shown as yellow circles in the T-X plot.

### F-X Bounding-Box Labeling Workflow

F-X bounding-box labeling uses `F`:

    F -> enter F-X box labeling mode
    F -> finish F-X box labeling and calculate global frequency range

Before entering F-X box mode, the annotator must select:

- An apex point.
- Two distance-side points.

Within F-X box labeling mode:

| Action | Function |
|---|---|
| Double-click in large F-X plot | Add a new bounding box |
| Ctrl + Click on a bounding box | Delete the selected bounding box |
| Drag a bounding box | Move the box |
| Drag a bounding-box handle | Resize the box |
| Space | Move to the next relevant F-X time window |
| F | Finish F-X box labeling |



F-X thumbnail border colors indicate labeling state:

| Border color | Meaning |
|---|---|
| Red | Currently selected F-X window with no bounding box |
| Green | Currently selected F-X window with at least one bounding box |
| Orange | Relevant F-X window not currently selected and not yet labeled |
| Light green | Relevant F-X window not currently selected but already labeled |
| No border | F-X window outside the selected whale-call time range |


### Text Display Changes

The text display panel includes an annotation prompt below Cursor Mode. It shows:

- Current annotation stage.
- Next required action.
- Validation errors.
- Selected apex, endpoint, distance, and F-X box information.
- Completion messages.

### Temporary T-X Plot Markers

The T-X plot uses:

| Marker | Meaning |
|---|---|
| Red star | Apex |
| Cyan X | Endpoint |
| Yellow circles | Distance boundary points |

Temporary T-X markers, F-X bounding boxes, and F-X thumbnail highlights are cleared when starting a new annotation, saving an annotation, or pressing `Esc`.