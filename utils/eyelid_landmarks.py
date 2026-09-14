from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EyelidLandmarkCurves:
    upper: tuple[int, ...]
    lower: tuple[int, ...]

    @property
    def corners(self) -> tuple[int, int]:
        return self.upper[0], self.upper[-1]

    @property
    def closure_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.upper[1:-1], self.lower[1:-1]))

    @property
    def all_ids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((*self.upper, *self.lower)))


# Use the upper contour and the verified bottom-lid contour. MediaPipe's
# image-space side names are opposite the subject-relative ARKit/MetaHuman blink
# names used by TopoRig.
BLINK_LEFT_EYELIDS = EyelidLandmarkCurves(
    upper=(263, 466, 388, 387, 386, 385, 384, 398, 362),
    lower=(263, 255, 339, 254, 253, 252, 256, 341, 362),
)
BLINK_RIGHT_EYELIDS = EyelidLandmarkCurves(
    upper=(33, 246, 161, 160, 159, 158, 157, 173, 133),
    lower=(33, 25, 110, 24, 23, 22, 26, 112, 133),
)

LEFT_EYE_REGION_ACTION_UNITS = (8, 10, 20, 22)
RIGHT_EYE_REGION_ACTION_UNITS = (9, 11, 21, 23)


EYELIDS_BY_ACTION_UNIT = {
    **{
        action_unit_id: BLINK_LEFT_EYELIDS
        for action_unit_id in LEFT_EYE_REGION_ACTION_UNITS
    },
    **{
        action_unit_id: BLINK_RIGHT_EYELIDS
        for action_unit_id in RIGHT_EYE_REGION_ACTION_UNITS
    },
}


def eyelid_landmarks_for_action_unit(action_unit_id: int) -> EyelidLandmarkCurves:
    try:
        return EYELIDS_BY_ACTION_UNIT[int(action_unit_id)]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(EYELIDS_BY_ACTION_UNIT))
        raise ValueError(
            f"No eyelid landmark curves are defined for AU{action_unit_id}; "
            f"supported action units: {supported}."
        ) from exc
