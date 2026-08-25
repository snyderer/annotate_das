## Changes Made So Far

### Apex Labeling Workflow

Apex labeling uses `A`:

    A -> select or edit apex
    A -> confirm apex

- Press `A` to enter apex labeling mode.
- Click the T-X plot to select the apex.
- The apex can be clicked repeatedly until correct.
- Press `A` again to confirm the apex.

### Hyperbola and Distance-to-Cable Fitting Workflow

Distance-to-cable hyperbola fitting uses `H`:

    H -> enter hyperbola fitting mode
    Move Distance to cable slider -> update hyperbola in real time
    H -> confirm distance-to-cable fit

- Hyperbola fitting requires a confirmed apex.
- Press `H` to enter hyperbola fitting mode.
- The Distance to cable slider is enabled during hyperbola fitting mode.
- The fitted hyperbola is displayed on the T-X plot and the apex hyperbola is anchored at the selected apex.
- The source location is modeled as a three-dimensional point at a perpendicular distance from a straight cable.
- Moving the Distance to cable slider updates the hyperbola in real time.
- Press `H` again to confirm the selected distance-to-cable value.
- When hyperbola fitting is confirmed, the apex is snapped to the nearest actual DAS channel.
- The selected Distance to cable value is saved with the annotation.

### Endpoint Labeling Workflow

Endpoint labeling uses `E`:

    E -> select or edit endpoint
    E -> confirm endpoint

- Endpoint labeling requires a confirmed apex and confirmed distance-to-cable fit.
- Press `E` to enter endpoint labeling mode.
- Click near the endpoint time in the T-X plot.
- Endpoint distance is constrained to the same cable channel as the apex.
- Duration is calculated automatically from apex time and endpoint time.
- The endpoint can be clicked repeatedly until correct.
- The endpoint can be dragged horizontally to refine endpoint time.
- An endpoint-anchored hyperbola is also displayed during endpoint labeling.
- Press `E` again to confirm the endpoint.

### Min-Max Distance Labeling Workflow

Distance labeling uses `D`:

    D -> enter distance-point selection mode
    Click -> select first distance point near apex hyperbola
    Click -> select second distance point near apex hyperbola
    Additional clicks -> keep only the latest two distance points
    D -> distance labeling complete

- Distance labeling requires a confirmed endpoint.
- Press `D` to enter distance-side point labeling mode.
- Click two points near the apex-anchored hyperbola in the T-X plot.
- Distance points are snapped to the apex hyperbola.
- If a click is too far from the apex hyperbola, it is rejected and the user is asked to click closer to the fitted curve.
- The two most recently clicked points are retained.
- Press `D` again to confirm that the currently retained two points define the distance range.
- After distance labeling is confirmed, the full hyperbola is replaced by only the apex-hyperbola segment between the two confirmed distance-side points.

### F-X Bounding-Box Labeling Workflow

F-X bounding-box labeling uses `F`:

    F -> enter F-X box labeling mode
    F -> finish F-X box labeling and calculate global frequency range


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
- The final saved annotation row contains global `f_min` and `f_max` values.

### Existing Label Display and Deletion

When `Show Existing Labels` is enabled:

- Labels without a saved Distance to cable value display as a red apex dot only.
- Labels with valid Distance to cable, Distance minimum, and Distance maximum values display:
  - A red apex dot.
  - A yellow dashed hyperbola segment.
  - Two red dots marking saved distance-side points.

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
| Cyan X | Current editable endpoint |
| Yellow circles | Current distance-side points |
| White dashed line | Apex-anchored hyperbola during fitting and distance selection |
| Cyan dashed line | Endpoint-anchored hyperbola during endpoint selection |
| White line segment | Confirmed apex-hyperbola segment between distance-side points |
| Red dots | Existing saved-label apex and distance-side points |
| Yellow dashed line | Existing saved-label hyperbola segment |

Temporary T-X markers, hyperbola overlays, F-X bounding boxes, and F-X thumbnail highlights are cleared when starting a new annotation, saving an annotation, or pressing `Esc`.