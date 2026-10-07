# -*- coding: utf-8 -*-
"""EFPA-PEE core implementation.

Cleaned from the authors' validated research notebook for reproducibility.
This module contains the frozen four-schedule encoder, blind decoder,
reversible PEE mappings, header handling, and evaluation helpers.
It intentionally contains no local/Google Drive paths and no notebook-only
diagnostic cells.
"""

import math
import time
import zlib

import numpy as np
from skimage.metrics import structural_similarity

SCHEDULE_TO_CODE = {
    ("2-BIN", "2-BIN"): 0,
    ("2-BIN", "4-BIN"): 1,
    ("4-BIN", "2-BIN"): 2,
    ("4-BIN", "4-BIN"): 3,
}
CODE_TO_SCHEDULE = {code: schedule for schedule, code in SCHEDULE_TO_CODE.items()}

FINAL_SCHEDULES_17 = [
    ("2-BIN", "2-BIN"),
    ("2-BIN", "4-BIN"),
    ("4-BIN", "2-BIN"),
    ("4-BIN", "4-BIN"),
]

def forward_map_2bin(e, bit=None):

    e = int(e)

    if e == -1:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=-1.")

        return -2 + int(bit)

    if e == 0:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=0.")

        return int(bit)

    if e <= -2:
        return e - 1

    if e >= 1:
        return e + 1

    raise ValueError("Unexpected prediction error.")


def inverse_map_2bin(e_prime):

    e_prime = int(e_prime)

    if e_prime == -2:
        return -1, 0

    if e_prime == -1:
        return -1, 1

    if e_prime == 0:
        return 0, 0

    if e_prime == 1:
        return 0, 1

    if e_prime <= -3:
        return e_prime + 1, None

    if e_prime >= 2:
        return e_prime - 1, None

    raise ValueError("Invalid transformed prediction error.")


def embed_pixel_2bin(x, pred, bit=None):

    x = int(x)
    pred = int(pred)

    e = x - pred

    if e in [-1, 0]:

        if bit not in [0, 1]:
            raise ValueError("Bit required.")

        e_prime = forward_map_2bin(e, bit)
        embedded = True

    else:

        e_prime = forward_map_2bin(e)
        embedded = False

    x_prime = pred + e_prime

    if not (0 <= x_prime <= 255):
        raise ValueError(
            f"Unsafe 2-bin pixel: x={x}, pred={pred}, "
            f"e={e}, e'={e_prime}, x'={x_prime}"
        )

    return int(x_prime), embedded


def extract_pixel_2bin(x_prime, pred):

    x_prime = int(x_prime)
    pred = int(pred)

    e_prime = x_prime - pred

    e, bit = inverse_map_2bin(e_prime)

    recovered_x = pred + e

    if not (0 <= recovered_x <= 255):
        raise ValueError("Invalid recovered 2-bin pixel.")

    return int(recovered_x), bit


def forward_map_4bin(e, bit=None):

    e = int(e)

    if e == -2:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=-2.")

        return -4 if bit == 0 else -3

    if e == -1:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=-1.")

        return -2 if bit == 0 else -1

    if e == 0:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=0.")

        return 0 if bit == 0 else 1

    if e == 1:

        if bit not in [0, 1]:
            raise ValueError("Bit required for e=1.")

        return 2 if bit == 0 else 3

    if e <= -3:
        return e - 2

    if e >= 2:
        return e + 2

    raise ValueError("Unexpected prediction error.")


def inverse_map_4bin(e_prime):

    e_prime = int(e_prime)

    if e_prime == -4:
        return -2, 0

    if e_prime == -3:
        return -2, 1

    if e_prime == -2:
        return -1, 0

    if e_prime == -1:
        return -1, 1

    if e_prime == 0:
        return 0, 0

    if e_prime == 1:
        return 0, 1

    if e_prime == 2:
        return 1, 0

    if e_prime == 3:
        return 1, 1

    if e_prime <= -5:
        return e_prime + 2, None

    if e_prime >= 4:
        return e_prime - 2, None

    raise ValueError("Invalid transformed 4-bin error.")


def embed_pixel_4bin(x, pred, bit=None):

    x = int(x)
    pred = int(pred)

    e = x - pred

    if e in [-2, -1, 0, 1]:

        if bit not in [0, 1]:
            raise ValueError("Bit required.")

        e_prime = forward_map_4bin(e, bit)
        embedded = True

    else:

        e_prime = forward_map_4bin(e)
        embedded = False

    x_prime = pred + e_prime

    if not (0 <= x_prime <= 255):
        raise ValueError(
            f"Unsafe 4-bin pixel: x={x}, pred={pred}, "
            f"e={e}, e'={e_prime}, x'={x_prime}"
        )

    return int(x_prime), embedded


def extract_pixel_4bin(x_prime, pred):

    x_prime = int(x_prime)
    pred = int(pred)

    e_prime = x_prime - pred

    e, bit = inverse_map_4bin(e_prime)

    recovered_x = pred + e

    if not (0 <= recovered_x <= 255):
        raise ValueError("Invalid recovered 4-bin pixel.")

    return int(recovered_x), bit


def directional_predict_common(img, i, j):

    left = int(img[i, j - 1])
    right = int(img[i, j + 1])
    up = int(img[i - 1, j])
    down = int(img[i + 1, j])

    P_H = (left + right) / 2.0
    P_V = (up + down) / 2.0

    d_H = abs(left - right)
    d_V = abs(up - down)

    if d_H <= d_V:
        pred = P_H
    else:
        pred = P_V

    return int(
        np.clip(
            round(pred),
            0,
            255
        )
    )


def local_complexity_common(img, i, j):

    left = int(img[i, j - 1])
    right = int(img[i, j + 1])
    up = int(img[i - 1, j])
    down = int(img[i + 1, j])

    return (
        abs(left - right)
        +
        abs(up - down)
        +
        abs(left - up)
        +
        abs(right - down)
    )


def checkerboard_coords_common(h, w, parity):

    if parity not in [0, 1]:
        raise ValueError("Parity must be 0 or 1.")

    return [
        (i, j)

        for i in range(2, h - 2)

        for j in range(2, w - 2)

        if (i + j) % 2 == parity
    ]


def protected_target_coords_common(h, w):

    return [
        (i, j)

        for i in range(2, h - 2)

        for j in range(2, w - 2)
    ]


def preprocess_boundaries_common(img):

    processed = img.copy().astype(np.uint8)

    h, w = processed.shape

    coords = protected_target_coords_common(
        h,
        w
    )

    location_map = []

    for i, j in coords:

        x = int(processed[i, j])

        if x == 0:

            processed[i, j] = 2
            location_map.append(1)

        elif x == 1:

            processed[i, j] = 3
            location_map.append(1)

        elif x == 254:

            processed[i, j] = 252
            location_map.append(1)

        elif x == 255:

            processed[i, j] = 253
            location_map.append(1)

        else:

            location_map.append(0)

    return processed, location_map


def restore_boundaries_common(
    recovered_preprocessed,
    location_map
):

    restored = (
        recovered_preprocessed
        .copy()
        .astype(np.uint8)
    )

    h, w = restored.shape

    coords = protected_target_coords_common(
        h,
        w
    )

    if len(coords) != len(location_map):
        raise ValueError(
            "Location-map length mismatch."
        )

    for (i, j), flag in zip(
        coords,
        location_map
    ):

        if int(flag) == 1:

            x = int(restored[i, j])

            if x == 2:
                restored[i, j] = 0

            elif x == 3:
                restored[i, j] = 1

            elif x == 252:
                restored[i, j] = 254

            elif x == 253:
                restored[i, j] = 255

            else:
                raise ValueError(
                    f"Invalid boundary restoration "
                    f"value {x} at {(i, j)}"
                )

    return restored


def sorted_checkerboard_coords_common(
    img,
    parity
):

    h, w = img.shape

    coords = checkerboard_coords_common(
        h,
        w,
        parity
    )

    scored = []

    for i, j in coords:

        c = local_complexity_common(
            img,
            i,
            j
        )

        scored.append(
            (c, i, j)
        )

    scored.sort(
        key=lambda z: (
            z[0],
            z[1],
            z[2]
        )
    )

    return [
        (i, j)
        for c, i, j in scored
    ]


def load_payload_bits_exact(path, N):

    raw = np.fromfile(
        path,
        dtype=np.uint8
    )

    if (
        len(raw) >= N
        and
        np.all(
            np.isin(
                raw[:N],
                [0, 1]
            )
        )
    ):

        bits = (
            raw[:N]
            .astype(int)
            .tolist()
        )

        storage = "0/1 bytes"

    else:

        bits = (
            np.unpackbits(
                raw
            )[:N]
            .astype(int)
            .tolist()
        )

        storage = "packed-bitstream"

    if len(bits) != N:

        raise RuntimeError(
            f"Only {len(bits)} bits available; "
            f"{N} requested."
        )

    return bits, storage


def estimate_stage_capacity_2bin(
    img,
    parity
):

    coords = sorted_checkerboard_coords_common(
        img,
        parity
    )

    capacity = 0

    for i, j in coords:

        pred = directional_predict_common(
            img,
            i,
            j
        )

        e = int(img[i, j]) - pred

        if e in [-1, 0]:
            capacity += 1

    return capacity, len(coords)


def embed_stage_2bin(
    img,
    payload_bits,
    parity
):

    marked = img.copy().astype(np.uint8)

    coords = sorted_checkerboard_coords_common(
        marked,
        parity
    )

    bit_index = 0
    processed_count = 0

    if len(payload_bits) == 0:

        return (
            marked,
            0,
            0,
            True
        )

    for i, j in coords:

        pred = directional_predict_common(
            marked,
            i,
            j
        )

        x = int(marked[i, j])

        e = x - pred

        if e in [-1, 0]:

            bit = int(
                payload_bits[
                    bit_index
                ]
            )

            x_prime, embedded = (
                embed_pixel_2bin(
                    x,
                    pred,
                    bit
                )
            )

            marked[i, j] = np.uint8(
                x_prime
            )

            bit_index += 1
            processed_count += 1

            if bit_index == len(
                payload_bits
            ):
                break

        else:

            x_prime, embedded = (
                embed_pixel_2bin(
                    x,
                    pred
                )
            )

            marked[i, j] = np.uint8(
                x_prime
            )

            processed_count += 1

    full_payload = (
        bit_index
        ==
        len(payload_bits)
    )

    return (
        marked,
        bit_index,
        processed_count,
        full_payload
    )


def extract_stage_2bin(
    marked,
    processed_count,
    parity
):

    recovered = (
        marked.copy()
        .astype(np.uint8)
    )

    coords = sorted_checkerboard_coords_common(
        recovered,
        parity
    )

    if (
        processed_count < 0
        or
        processed_count > len(coords)
    ):
        raise ValueError(
            "Invalid 2-bin processed_count."
        )

    processed_coords = (
        coords[:processed_count]
    )

    extracted_reversed = []

    for i, j in reversed(
        processed_coords
    ):

        pred = directional_predict_common(
            recovered,
            i,
            j
        )

        x_prime = int(
            recovered[i, j]
        )

        original_x, bit = (
            extract_pixel_2bin(
                x_prime,
                pred
            )
        )

        recovered[i, j] = np.uint8(
            original_x
        )

        if bit is not None:

            extracted_reversed.append(
                int(bit)
            )

    return (
        recovered,
        extracted_reversed[::-1]
    )


def estimate_stage_capacity_4bin(
    img,
    parity
):

    coords = sorted_checkerboard_coords_common(
        img,
        parity
    )

    capacity = 0

    for i, j in coords:

        pred = directional_predict_common(
            img,
            i,
            j
        )

        e = int(img[i, j]) - pred

        if e in [-2, -1, 0, 1]:
            capacity += 1

    return capacity, len(coords)


def embed_stage_4bin(
    img,
    payload_bits,
    parity
):

    marked = img.copy().astype(np.uint8)

    coords = sorted_checkerboard_coords_common(
        marked,
        parity
    )

    bit_index = 0
    processed_count = 0

    if len(payload_bits) == 0:

        return (
            marked,
            0,
            0,
            True
        )

    for i, j in coords:

        pred = directional_predict_common(
            marked,
            i,
            j
        )

        x = int(
            marked[i, j]
        )

        e = x - pred

        if e in [-2, -1, 0, 1]:

            bit = int(
                payload_bits[
                    bit_index
                ]
            )

            x_prime, embedded = (
                embed_pixel_4bin(
                    x,
                    pred,
                    bit
                )
            )

            marked[i, j] = np.uint8(
                x_prime
            )

            bit_index += 1
            processed_count += 1

            if bit_index == len(
                payload_bits
            ):
                break

        else:

            x_prime, embedded = (
                embed_pixel_4bin(
                    x,
                    pred
                )
            )

            marked[i, j] = np.uint8(
                x_prime
            )

            processed_count += 1

    full_payload = (
        bit_index
        ==
        len(payload_bits)
    )

    return (
        marked,
        bit_index,
        processed_count,
        full_payload
    )


def extract_stage_4bin(
    marked,
    processed_count,
    parity
):

    recovered = (
        marked.copy()
        .astype(np.uint8)
    )

    coords = sorted_checkerboard_coords_common(
        recovered,
        parity
    )

    if (
        processed_count < 0
        or
        processed_count > len(coords)
    ):
        raise ValueError(
            "Invalid 4-bin processed_count."
        )

    processed_coords = (
        coords[:processed_count]
    )

    extracted_reversed = []

    for i, j in reversed(
        processed_coords
    ):

        pred = directional_predict_common(
            recovered,
            i,
            j
        )

        x_prime = int(
            recovered[i, j]
        )

        original_x, bit = (
            extract_pixel_4bin(
                x_prime,
                pred
            )
        )

        recovered[i, j] = np.uint8(
            original_x
        )

        if bit is not None:

            extracted_reversed.append(
                int(bit)
            )

    return (
        recovered,
        extracted_reversed[::-1]
    )


def diagnostic_quality(
    original_img,
    marked_img
):

    diff = (
        marked_img.astype(np.float64)
        -
        original_img.astype(np.float64)
    )

    sse = float(
        np.sum(
            diff ** 2
        )
    )

    mse = float(
        np.mean(
            diff ** 2
        )
    )

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = float(
            10.0
            *
            np.log10(
                (255.0 ** 2)
                /
                mse
            )
        )

    modified = int(
        np.count_nonzero(
            marked_img
            !=
            original_img
        )
    )

    max_change = int(
        np.max(
            np.abs(
                marked_img.astype(np.int16)
                -
                original_img.astype(np.int16)
            )
        )
    )

    return {
        "SSE": sse,
        "MSE": mse,
        "PSNR": psnr,
        "Modified": modified,
        "Max_change": max_change
    }


def int_to_bits_scdm(value, width):

    value = int(value)

    if value < 0 or value >= (1 << width):

        raise ValueError(
            f"Value {value} does not fit "
            f"in {width} bits."
        )

    return [

        (value >> k) & 1

        for k in range(
            width - 1,
            -1,
            -1
        )
    ]


def bits_to_int_scdm(bits):

    value = 0

    for b in bits:

        value = (
            (value << 1)
            |
            int(b)
        )

    return value


def bytes_to_bits_scdm(data):

    if len(data) == 0:

        return []

    return (
        np.unpackbits(
            np.frombuffer(
                data,
                dtype=np.uint8
            )
        )
        .astype(int)
        .tolist()
    )


def bits_to_bytes_scdm(bits):

    if len(bits) == 0:

        return b""

    if len(bits) % 8 != 0:

        raise ValueError(
            "Byte conversion requires "
            "a multiple of 8 bits."
        )

    return (
        np.packbits(
            np.asarray(
                bits,
                dtype=np.uint8
            )
        )
        .tobytes()
    )


def compress_lm_scdm(location_map):

    raw = np.packbits(

        np.asarray(
            location_map,
            dtype=np.uint8
        )
    ).tobytes()

    return zlib.compress(raw)


def decompress_lm_scdm(
    compressed_bytes,
    lm_length
):

    raw = zlib.decompress(
        compressed_bytes
    )

    bits = (
        np.unpackbits(
            np.frombuffer(
                raw,
                dtype=np.uint8
            )
        )
        .astype(int)
        .tolist()
    )

    if len(bits) < lm_length:

        raise RuntimeError(
            "Decompressed location map "
            "is too short."
        )

    return bits[:lm_length]


def schedule_header_widths_scdm(
    h,
    w
):

    protected_count = len(

        protected_target_coords_common(
            h,
            w
        )
    )


    stage_count = max(

        len(
            checkerboard_coords_common(
                h,
                w,
                0
            )
        ),

        len(
            checkerboard_coords_common(
                h,
                w,
                1
            )
        )
    )


    N_bits = max(

        1,

        math.ceil(
            math.log2(
                protected_count
                +
                1
            )
        )
    )


    K_bits = max(

        1,

        math.ceil(
            math.log2(
                stage_count
                +
                1
            )
        )
    )


    # Raw packed location-map bytes
    max_raw_lm_bytes = (

        math.ceil(
            protected_count
            /
            8
        )
    )


    # Safe field range for compressed stream.
    max_lm_bytes = (

        max_raw_lm_bytes
        +
        1025
    )


    LM_bits = max(

        1,

        math.ceil(
            math.log2(
                max_lm_bytes
                +
                1
            )
        )
    )


    total = (

        2       # schedule ID

        +

        N_bits

        +

        K_bits

        +

        K_bits

        +

        LM_bits
    )


    return {

        "schedule_bits":
            2,

        "N_bits":
            N_bits,

        "K1_bits":
            K_bits,

        "K2_bits":
            K_bits,

        "LM_bits":
            LM_bits,

        "header_bits":
            total
    }


def pack_schedule_header_scdm(
    schedule,
    N,
    K1,
    K2,
    lm_bytes,
    h,
    w
):

    widths = (
        schedule_header_widths_scdm(
            h,
            w
        )
    )


    code = (
        SCHEDULE_TO_CODE[
            tuple(schedule)
        ]
    )


    bits = []


    bits += int_to_bits_scdm(
        code,
        2
    )


    bits += int_to_bits_scdm(
        N,
        widths[
            "N_bits"
        ]
    )


    bits += int_to_bits_scdm(
        K1,
        widths[
            "K1_bits"
        ]
    )


    bits += int_to_bits_scdm(
        K2,
        widths[
            "K2_bits"
        ]
    )


    bits += int_to_bits_scdm(
        lm_bytes,
        widths[
            "LM_bits"
        ]
    )


    if len(bits) != widths[
        "header_bits"
    ]:

        raise RuntimeError(
            "Header packing length error."
        )


    return bits


def unpack_schedule_header_scdm(
    bits,
    h,
    w
):

    widths = (
        schedule_header_widths_scdm(
            h,
            w
        )
    )


    if len(bits) != widths[
        "header_bits"
    ]:

        raise ValueError(
            "Incorrect schedule-header length."
        )


    p = 0


    schedule_code = (
        bits_to_int_scdm(
            bits[
                p:
                p + 2
            ]
        )
    )

    p += 2


    if schedule_code not in (
        CODE_TO_SCHEDULE
    ):

        raise ValueError(
            "Invalid schedule code."
        )


    N = bits_to_int_scdm(

        bits[
            p:
            p
            +
            widths[
                "N_bits"
            ]
        ]
    )

    p += widths[
        "N_bits"
    ]


    K1 = bits_to_int_scdm(

        bits[
            p:
            p
            +
            widths[
                "K1_bits"
            ]
        ]
    )

    p += widths[
        "K1_bits"
    ]


    K2 = bits_to_int_scdm(

        bits[
            p:
            p
            +
            widths[
                "K2_bits"
            ]
        ]
    )

    p += widths[
        "K2_bits"
    ]


    lm_bytes = bits_to_int_scdm(

        bits[
            p:
            p
            +
            widths[
                "LM_bits"
            ]
        ]
    )


    return {

        "Schedule":
            CODE_TO_SCHEDULE[
                schedule_code
            ],

        "N":
            N,

        "K1":
            K1,

        "K2":
            K2,

        "LM_bytes":
            lm_bytes
    }


def write_schedule_header_scdm(
    img,
    header_bits
):

    out = (
        img.copy()
        .astype(np.uint8)
    )


    h, w = out.shape


    if len(header_bits) > w:

        raise RuntimeError(
            "Header does not fit "
            "in the first row."
        )


    saved = []


    for j, bit in enumerate(
        header_bits
    ):

        x = int(
            out[
                0,
                j
            ]
        )


        saved.append(
            x & 1
        )


        out[
            0,
            j
        ] = np.uint8(

            (x & 254)

            |

            int(bit)
        )


    return (
        out,
        saved
    )


def read_schedule_header_scdm(
    stego
):

    h, w = stego.shape


    H = (
        schedule_header_widths_scdm(
            h,
            w
        )[
            "header_bits"
        ]
    )


    bits = [

        int(
            stego[
                0,
                j
            ]
        )
        &
        1

        for j in range(H)
    ]


    return (
        bits,
        unpack_schedule_header_scdm(
            bits,
            h,
            w
        )
    )


def restore_schedule_header_scdm(
    img,
    saved_lsbs
):

    out = (
        img.copy()
        .astype(np.uint8)
    )


    for j, bit in enumerate(
        saved_lsbs
    ):

        x = int(
            out[
                0,
                j
            ]
        )


        out[
            0,
            j
        ] = np.uint8(

            (x & 254)

            |

            int(bit)
        )


    return out


def stage_capacity_scdm(
    img,
    parity,
    mode
):

    if mode == "2-BIN":

        return (
            estimate_stage_capacity_2bin(
                img,
                parity
            )
        )


    elif mode == "4-BIN":

        return (
            estimate_stage_capacity_4bin(
                img,
                parity
            )
        )


    raise ValueError(
        "Unknown mode."
    )


def stage_embed_scdm(
    img,
    bits,
    parity,
    mode
):

    if mode == "2-BIN":

        return embed_stage_2bin(

            img,
            bits,
            parity
        )


    elif mode == "4-BIN":

        return embed_stage_4bin(

            img,
            bits,
            parity
        )


    raise ValueError(
        "Unknown mode."
    )


def stage_extract_scdm(
    img,
    K,
    parity,
    mode
):

    if mode == "2-BIN":

        return extract_stage_2bin(

            img,
            K,
            parity
        )


    elif mode == "4-BIN":

        return extract_stage_4bin(

            img,
            K,
            parity
        )


    raise ValueError(
        "Unknown mode."
    )


def embed_schedule_scdm(
    preprocessed,
    internal_bits,
    schedule
):

    mode1, mode2 = schedule


    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    (
        C1,
        candidates1
    ) = stage_capacity_scdm(

        preprocessed,
        0,
        mode1
    )


    (
        stage1_marked,
        E1,
        K1,
        full1
    ) = stage_embed_scdm(

        preprocessed,
        internal_bits,
        0,
        mode1
    )


    remaining = (
        internal_bits[
            E1:
        ]
    )


    # --------------------------------------------------------
    # Stage 2 capacity MUST be evaluated on Stage-1 state.
    # --------------------------------------------------------

    (
        C2,
        candidates2
    ) = stage_capacity_scdm(

        stage1_marked,
        1,
        mode2
    )


    if len(
        remaining
    ) > 0:

        (
            stage2_marked,
            E2,
            K2,
            full2
        ) = stage_embed_scdm(

            stage1_marked,
            remaining,
            1,
            mode2
        )

    else:

        stage2_marked = (
            stage1_marked.copy()
        )

        E2 = 0

        K2 = 0

        full2 = True


    total_embedded = (
        E1
        +
        E2
    )


    full_stream = bool(

        total_embedded

        ==

        len(
            internal_bits
        )
    )


    return {

        "Marked":
            stage2_marked,

        "Stage1_capacity":
            int(C1),

        "Stage2_capacity":
            int(C2),

        "Stage1_embedded":
            int(E1),

        "Stage2_embedded":
            int(E2),

        "K1":
            int(K1),

        "K2":
            int(K2),

        "Total_embedded":
            int(total_embedded),

        "Full_stream":
            full_stream
    }


def blind_decode_schedule_scdm(
    stego
):

    h, w = stego.shape


    header_bits, metadata = (
        read_schedule_header_scdm(
            stego
        )
    )


    schedule = tuple(
        metadata[
            "Schedule"
        ]
    )


    mode1, mode2 = (
        schedule
    )


    N = int(
        metadata[
            "N"
        ]
    )


    K1 = int(
        metadata[
            "K1"
        ]
    )


    K2 = int(
        metadata[
            "K2"
        ]
    )


    LM_bytes = int(
        metadata[
            "LM_bytes"
        ]
    )


    H = len(
        header_bits
    )


    # --------------------------------------------------------
    # Reverse Stage 2 FIRST
    # --------------------------------------------------------

    after_stage2_reverse, bits2 = (
        stage_extract_scdm(

            stego,

            K2,

            1,

            mode2
        )
    )


    # --------------------------------------------------------
    # Reverse Stage 1 SECOND
    # --------------------------------------------------------

    recovered_preprocessed, bits1 = (
        stage_extract_scdm(

            after_stage2_reverse,

            K1,

            0,

            mode1
        )
    )


    extracted_internal = (

        bits1
        +
        bits2
    )


    lm_bit_length = (
        LM_bytes
        *
        8
    )


    expected_length = (

        N

        +
        H

        +
        lm_bit_length
    )


    if len(
        extracted_internal
    ) != expected_length:

        raise RuntimeError(

            "Internal stream length mismatch: "

            f"{len(extracted_internal)} "

            f"vs "

            f"{expected_length}"
        )


    p = 0


    payload = (
        extracted_internal[
            p:
            p + N
        ]
    )

    p += N


    saved_header_lsbs = (
        extracted_internal[
            p:
            p + H
        ]
    )

    p += H


    recovered_lm_bits = (
        extracted_internal[
            p:
            p + lm_bit_length
        ]
    )


    compressed_lm = (
        bits_to_bytes_scdm(
            recovered_lm_bits
        )
    )


    lm_length = len(

        protected_target_coords_common(
            h,
            w
        )
    )


    recovered_lm = (
        decompress_lm_scdm(

            compressed_lm,

            lm_length
        )
    )


    # --------------------------------------------------------
    # Restore original first-row LSBs
    # --------------------------------------------------------

    recovered_preprocessed = (
        restore_schedule_header_scdm(

            recovered_preprocessed,

            saved_header_lsbs
        )
    )


    # --------------------------------------------------------
    # Restore boundary preprocessing
    # --------------------------------------------------------

    recovered_original = (
        restore_boundaries_common(

            recovered_preprocessed,

            recovered_lm
        )
    )


    return {

        "Schedule":
            schedule,

        "N":
            N,

        "K1":
            K1,

        "K2":
            K2,

        "LM_bytes":
            LM_bytes,

        "Payload":
            payload,

        "Location_map":
            recovered_lm,

        "Recovered_original":
            recovered_original
    }


def schedule_label_17(schedule):

    return (
        schedule[0].replace("-BIN", "")
        +
        "->"
        +
        schedule[1].replace("-BIN", "")
    )


def ssim_17(original, stego):

    return float(
        structural_similarity(
            original,
            stego,
            data_range=255
        )
    )


def ber_17(a, b):

    if len(a) != len(b):

        return np.nan, -1

    aa = np.asarray(
        a,
        dtype=np.uint8
    )

    bb = np.asarray(
        b,
        dtype=np.uint8
    )

    errors = int(
        np.count_nonzero(
            aa != bb
        )
    )

    return (
        float(
            errors / len(aa)
        )
        if len(aa) > 0
        else 0.0,
        errors
    )


def max_recovery_error_17(a, b):

    return int(
        np.max(
            np.abs(
                a.astype(np.int16)
                -
                b.astype(np.int16)
            )
        )
    )


def prepare_case_17(
    original,
    payload_bits
):

    original = (
        original.copy()
        .astype(np.uint8)
    )

    h, w = original.shape

    N = len(payload_bits)


    # --------------------------------------------------------
    # Reversible boundary preprocessing
    # --------------------------------------------------------

    (
        preprocessed,
        location_map
    ) = preprocess_boundaries_common(
        original
    )


    # --------------------------------------------------------
    # Compress location map
    # --------------------------------------------------------

    compressed_lm = (
        compress_lm_scdm(
            location_map
        )
    )


    lm_bits = (
        bytes_to_bits_scdm(
            compressed_lm
        )
    )


    LM_bytes = len(
        compressed_lm
    )


    # --------------------------------------------------------
    # Blind-header width
    # --------------------------------------------------------

    widths = (
        schedule_header_widths_scdm(
            h,
            w
        )
    )


    H = int(
        widths[
            "header_bits"
        ]
    )


    if H > w:

        raise RuntimeError(
            "Blind header does not fit "
            "in the first image row."
        )


    # --------------------------------------------------------
    # Save original LSBs overwritten by blind header
    # --------------------------------------------------------

    saved_header_lsbs = [

        int(
            original[
                0,
                j
            ]
        )
        & 1

        for j in range(H)
    ]


    # --------------------------------------------------------
    # Complete PEE-embedded reversible stream
    #
    # B = payload || saved-header-LSBs || compressed-LM
    # --------------------------------------------------------

    internal_bits = (

        list(payload_bits)

        +

        saved_header_lsbs

        +

        lm_bits
    )


    internal_aux_bits = (

        H
        +
        len(lm_bits)
    )


    # Blind header itself is separate from the PEE stream.
    #
    # Therefore:
    #
    # internal_aux_bits =
    #       saved header LSBs + compressed LM
    #
    # total_side_information_bits =
    #       internal auxiliary bits + blind header metadata
    #
    total_side_information_bits = (

        internal_aux_bits
        +
        H
    )


    return {

        "Original":
            original,

        "Preprocessed":
            preprocessed,

        "Location_map":
            location_map,

        "Compressed_LM":
            compressed_lm,

        "LM_bits":
            lm_bits,

        "LM_bytes":
            LM_bytes,

        "Header_bits":
            H,

        "Saved_header_lsbs":
            saved_header_lsbs,

        "Internal_bits_list":
            internal_bits,

        "Internal_bits":
            len(internal_bits),

        "Internal_aux_bits":
            internal_aux_bits,

        "Total_side_information_bits":
            total_side_information_bits,

        "Boundary_changed":
            int(
                sum(location_map)
            ),

        "N":
            N,

        "Height":
            h,

        "Width":
            w
    }


def encode_candidate_17(
    context,
    schedule
):

    original = context[
        "Original"
    ]

    preprocessed = context[
        "Preprocessed"
    ]

    internal_bits = context[
        "Internal_bits_list"
    ]

    N = context[
        "N"
    ]

    H = context[
        "Header_bits"
    ]

    LM_bytes = context[
        "LM_bytes"
    ]

    saved_header_lsbs = context[
        "Saved_header_lsbs"
    ]

    h = context[
        "Height"
    ]

    w = context[
        "Width"
    ]


    # --------------------------------------------------------
    # Two-stage state-coupled embedding
    # --------------------------------------------------------

    stage = (
        embed_schedule_scdm(

            preprocessed,

            internal_bits,

            schedule
        )
    )


    result = {

        "Schedule":
            tuple(schedule),

        "Schedule_label":
            schedule_label_17(
                schedule
            ),

        "Feasible":
            bool(
                stage[
                    "Full_stream"
                ]
            ),

        "Stage_info":
            stage,

        "SSE":
            np.nan,

        "Stego":
            None,

        "Header":
            None
    }


    # --------------------------------------------------------
    # Infeasible schedule
    # --------------------------------------------------------

    if not result[
        "Feasible"
    ]:

        return result


    # --------------------------------------------------------
    # Construct exact blind header
    # --------------------------------------------------------

    header = (
        pack_schedule_header_scdm(

            schedule,

            N,

            stage[
                "K1"
            ],

            stage[
                "K2"
            ],

            LM_bytes,

            h,

            w
        )
    )


    if len(header) != H:

        raise RuntimeError(
            "Header-length mismatch."
        )


    # --------------------------------------------------------
    # Write blind header
    # --------------------------------------------------------

    (
        stego,
        overwritten_lsbs
    ) = write_schedule_header_scdm(

        stage[
            "Marked"
        ],

        header
    )


    # First row is outside the two-pixel embedding region,
    # so these saved LSBs must agree exactly.
    if (
        overwritten_lsbs
        !=
        saved_header_lsbs
    ):

        raise RuntimeError(
            "Header-LSB consistency failure."
        )


    # --------------------------------------------------------
    # Exact final SSE used for candidate selection
    # --------------------------------------------------------

    d = (

        stego.astype(
            np.int64
        )

        -

        original.astype(
            np.int64
        )
    )


    final_sse = int(
        np.sum(
            d * d
        )
    )


    result.update({

        "SSE":
            final_sse,

        "Stego":
            stego,

        "Header":
            header
    })


    return result


def final_encode_17(
    original,
    payload_bits
):

    t0 = (
        time.perf_counter()
    )


    context = (
        prepare_case_17(
            original,
            payload_bits
        )
    )


    candidates = {}


    for schedule in FINAL_SCHEDULES_17:

        candidate = (
            encode_candidate_17(
                context,
                schedule
            )
        )

        candidates[
            tuple(schedule)
        ] = candidate


    feasible = [

        r

        for r in candidates.values()

        if r[
            "Feasible"
        ]
    ]


    if len(feasible) == 0:

        selected = None

    else:

        selected = min(

            feasible,

            key=lambda r: (

                int(
                    r[
                        "SSE"
                    ]
                ),

                SCHEDULE_TO_CODE[
                    tuple(
                        r[
                            "Schedule"
                        ]
                    )
                ]
            )
        )


    encoding_seconds = (

        time.perf_counter()
        -
        t0
    )


    return {

        "Context":
            context,

        "Candidates":
            candidates,

        "Selected":
            selected,

        "Encoding_seconds":
            float(
                encoding_seconds
            )
    }


