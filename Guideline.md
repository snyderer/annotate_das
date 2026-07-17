## Changes Made So Far

### Apex-Duration Labeling Workflow

Apex and endpoint labeling uses `A`:

    A -> select or edit apex
    A -> select or edit endpoint
    A -> apex and endpoint complete

- Apex / endpoint can be clicked repeatedly until correct.
- Pressing `A` explicitly advances the stage.
- Duration is calculated from apex and endpoint time.
- The apex can be dragged after selection.
- Dragging the apex moves the endpoint and fitted hyperbola together.

### Hyperbola and Distance-to-Cable Fitting Workflow

Distance-to-cable hyperbola fitting uses `H`:

    H -> enter hyperbola fitting mode
    Move Distance to cable slider -> update hyperbola in real time
    H -> finish hyperbola fitting

- Hyperbola fitting requires an apex point.
- The Distance to cable slider is enabled during hyperbola fitting mode.
- The fitted hyperbola is displayed on the T-X plot and the hyperbola is anchored at the selected apex.
- The source location is modeled as a three-dimensional point at a perpendicular distance from a straight cable.
- The selected Distance to cable value is saved with the annotation.
- The apex marker may be dragged to adjust the fitted hyperbola position.
- Dragging the apex updates the apex time and distance, moves the endpoint with the apex, and redraws the hyperbola.

### Min-Max Distance Labeling Workflow

Distance labeling uses `D`:

    D -> enter distance-point selection mode
    Click -> select first distance point near the fitted hyperbola
    Click -> select second distance point near the fitted hyperbola
    Additional clicks -> keep only the latest two distance points
    D -> distance labeling complete

- Press `D` after apex, endpoint, and hyperbola fitting are complete.
- Click two points near the fitted hyperbola on the T-X plot.
- If a click is too far from the hyperbola, it is rejected and the user is asked to click closer to the fitted curve.
- The two most recently clicked points are retained.
- Distance minimum and maximum are calculated automatically from the two retained points.
- Press `D` again to confirm that the currently retained two points define the distance range.
- The two selected points are shown as yellow circles in the T-X plot.
- After distance labeling is confirmed, the full fitted hyperbola is replaced by only the hyperbola segment between the two confirmed distance points.

### F-X Bounding-Box Labeling Workflow

F-X bounding-box labeling uses `F`:

    F -> enter F-X box labeling mode
    F -> finish F-X box labeling and calculate global frequency range

Before entering F-X box mode, the annotator must select:

- An apex point.
- Two confirmed distance-side points.

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

When F-X box labeling is finished:

- All manually created F-X boxes are collected.
- Frequency minimum is calculated from the lowest frequency boundary across all boxes.
- Frequency maximum is calculated from the highest frequency boundary across all boxes.
- The final saved annotation row contains the global `f_min` and `f_max` values.

### Existing Label Display and Deletion

When `Show Existing Labels` is enabled:

- Labels without a saved Distance to cable value display as a red apex dot only.
- Labels with valid Distance to cable, Distance minimum, and Distance maximum values display:
  - A red apex dot.
  - A yellow dashed hyperbola segment.
  - Two red dots marking the saved distance-side points.

To delete an existing label:

    Ctrl + Click near an existing red apex dot

- A confirmation dialog appears before deletion.
- Selecting `Yes` permanently removes the label from the CSV file.
- Selecting `No` cancels deletion.
- The existing-label display refreshes after deletion.

### Text Display Changes

The text display panel includes an annotation prompt below Cursor Mode. It shows:

- Current annotation stage.
- Next required action.
- Validation errors.
- Selected apex, endpoint, distance, hyperbola, and F-X box information.
- Completion messages.

### Temporary T-X Plot Markers

The T-X plot uses:

| Marker | Meaning |
|---|---|
| Red star | Current editable apex |
| Cyan X | Current endpoint |
| Yellow circles | Current distance-side points |
| White dashed line | Active hyperbola during fitting and distance-point selection |
| White line segment | Confirmed hyperbola segment between distance-side points |
| Red dots | Existing saved-label apex and distance-side points |
| Yellow dashed line | Existing saved-label hyperbola segment |

Temporary T-X markers, active hyperbola overlays, F-X bounding boxes, and F-X thumbnail highlights are cleared when starting a new annotation, saving an annotation, or pressing `Esc`.