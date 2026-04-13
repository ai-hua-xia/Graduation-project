# Centered Throttle Overlay Design

## Goal

Update the left-bottom control panel so the second action dimension is easier to read visually. Instead of a left-to-right fill bar, the panel should render a centered bidirectional control bar whose zero point corresponds to raw value `0.40`.

## Scope

This change only affects the overlay visualization. It does not change model inputs, dataset values, training behavior, or action semantics.

The centered control bar should replace the current bottom bar everywhere the shared `draw_control_overlay(...)` helper is used, including:

- reference-video overlay outputs
- mixed reference/prediction outputs
- pure prediction outputs that reuse the same helper

## Current Problem

The existing throttle/control bar maps raw values into a one-sided fill from left to right. Because the current data range is narrow and mostly positive (`~0.28` to `~0.52`), values such as `0.42` and `0.50` do not read as clearly distinct control states. The user wants to interpret the bar relative to a neutral midpoint rather than as raw fill length.

## Proposed Design

Keep the current panel placement and steering display, but replace the bottom bar with a centered bidirectional control bar:

- raw value `0.40` maps to the visual midpoint and is labeled as `0`
- values above `0.40` extend to the right
- values below `0.40` extend to the left
- larger positive deltas appear farther right than smaller positive deltas
- larger negative deltas appear farther left than smaller negative deltas

The text display should show:

- a primary signed control delta, e.g. `control +0.07`
- a smaller raw value label, e.g. `raw 0.47`
- both values formatted with two decimals

## Visual Details

- keep the overlay in the same left-bottom panel
- keep the steering bar unchanged
- draw a center marker for the control bar
- add readable tick labels at `-0.10`, `0`, and `+0.10`
- use different colors for negative and positive fill directions
- do not add any new top-left watermark or source label

## Mapping Rule

The displayed control delta is:

`delta = raw_value - 0.40`

For rendering:

- center corresponds to `delta = 0`
- the bar display range is fixed to `[-0.12, +0.12]`
- the left edge corresponds to `delta = -0.12`
- the right edge corresponds to `delta = +0.12`
- tick labels at `-0.10`, `0`, and `+0.10` are internal reference ticks, not bar endpoints

The implementation should clamp out-of-range values into `[-0.12, +0.12]` so the indicator and fill stay within the bar.

## Testing

Add focused tests that verify:

- a larger positive raw value renders farther right than a smaller positive raw value
- a raw value below `0.40` renders to the left side of the center marker
- an out-of-range raw value is clamped to the bar edge rather than drawing outside the bar
- the overlay still renders in the left-bottom panel without introducing a top-left label
