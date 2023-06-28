# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def print_mask64(number, show_index=False):
    """Prints a 64-bit len3_lut input mask.

    In the Snappy 2-to-3 bytes optimization, the inputs are grouped in
    groups of ten, and then put into the len3_lut lookup table.
    Depending on the value in the lookup table, 8 (all symbols have 2 bytes) to
    12 bits (all symbols have 3 bytes) of the mask are consumed.

    Args:
        number (int): _description_
        show_index (bool, optional): Show the group and element-of-group index. Defaults to False.
    """
    tens = "".join((str(i // 10) for i in range(63, -1, -1)))
    ones = "".join((str(i % 10) for i in range(63, -1, -1)))
    ws = " " * (64 + 7)
    result = [
        " ".join((tens[i * 8 : i * 8 + 8] for i in range(0, 8))),
        " ".join((ones[i * 8 : i * 8 + 8] for i in range(0, 8))),
        " " * (64 + 7),
        " ".join([f"{number:064b}"[i * 8 : i * 8 + 8] for i in range(0, 8)]),
        (" " * 7).join([f"{number:016X}"[i * 2 : i * 2 + 2] for i in range(0, 8)]),
    ]
    if not show_index:
        result = result[2:]
    print("\n".join(result))


def to_bitsequence_64(numbers: list):
    """Emits a sequence of '0' and '1' chars that encode the bits of the passed 64-bit numbers.

    Emits a sequence of '0' and '1' chars that encode the bits of the passed numbers.
    First entry in sequence represents the highest bit of the last number's entry.
    Last entry in sequence represents the lowest bit of the first number's entry.
    """
    result = ""
    for number in numbers:
        result = f"{number:064b}" + result
    return result


def len3_mask(bitsequence, limit=64):
    """Computes the len 3 mask from a Big-Endian bitsequence and cuts it off at the specified limit.

    The bitsequence must be supplied as Big-Endian character sequence that contains
    '0' and '1'. Big-Endian means here that the last character represents the
    most-significant bit of the number the bitsequence has been generated from.

    The len3 mask is computed as follows:

    * Given an input sequence of '0' or '1', we place a '0' into the output
      sequence if the input sequence has a '0' at the cursor.
      In this case, we move the cursor by 2 steps (which
      implies that the respective Snappy symbol consumes 2 bytes).
    * If there is a '1', we place a '1' into the output stream
      and we move the cursors by 3 steps (which
      implies that the respective Snappy symbol consumes 3 bytes).

    The result is returned as character sequence in Little-Endian format (type `str`).
    """
    result = ""
    cursor = len(bitsequence) - 1  # start at last element
    orig_cursor = cursor
    while len(result) < limit:
        if bitsequence[cursor] == "1":  # indicates 3-byte symbol
            cursor -= 3
            result = "1" + result
        else:
            cursor -= 2
            result = "0" + result
        if cursor < 0:
            raise RuntimeError(
                "ERROR: cursor must not be negative. Input sequence to short?"
            )
    consumed_bits = orig_cursor - cursor
    return result, consumed_bits


if __name__ == "__main__":
    # Batch Test 2: cur_t: 141

    v0 = 1297045497366774530
    v1 = 306244775751782400
    v2 = 9800413331314508032

    m, _ = len3_mask(to_bitsequence_64([v0, v1, v2]))

    assert (
        int(m, 2) == 0b0000000000000000000000000000000000100000000000010000000000010000
    )

    # Batch Test 2: cur_t: 172
    v0 = 1225051666414313504
    v1 = 1157429502285062340
    v2 = 9223381383253332224

    m, _ = len3_mask(to_bitsequence_64([v0, v1, v2]))

    assert int(m, 2) == 4432414638080

    print_mask64(int(m, 2))
    print_mask64(1134696008253440)

    # app input 2: cur_t: 0
    # note: highest mask bit must be set in this case
    print("app input 2")
    v0 = 15995499993133274059
    v1 = 17575218449657001922
    v2 = 13172742451556175867

    m, consumed_bits = len3_mask(to_bitsequence_64([v0, v1, v2]))

    print(int(m, 2))
    print(consumed_bits)

    print_mask64(int(m, 2))
    print_mask64(8933997687287561919)

    # Generate additional test data for C++ test
    import random

    random.seed(0)

    for i in range(0, 5):
        v0 = random.randint(0, 2**64 - 1)
        v1 = random.randint(0, 2**64 - 1)
        v2 = random.randint(0, 2**64 - 1)

        m, _ = len3_mask(to_bitsequence_64([v0, v1, v2]))
        print(f"// random values test {i}")
        print(f"v0={v0}lu;")
        print(f"v1={v1}lu;")
        print(f"v2={v2}lu;")
        print(f"len3_mask_expected = {int(m,2)}lu;")
        print(
            f"""\
len3_mask = get_len3_mask(v0,v1,v2);
std::cout << len3_mask << std::endl;
 
if ( len3_mask != len3_mask_expected ) {{
  std::cout << "random values test {i}: FAIL" << std::endl;
}}
"""
        )
