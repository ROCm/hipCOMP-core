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

import logging

_log = logging.getLogger(__name__)

def decode_symbols(comp_stream,uncompressed_len,offset=0):
    """Returns a list of triples with symbol cursor, num bytes, and the symbol.

    Args:
        comp_stream (_type_): _description_
        uncompressed_len (_type_): _description_
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        `list`:
            A list of tuples of size 4 that each contains 
            (1) symbol cursor position, (2) the number of bytes to encode the symbol, (3) the output put stream destination, (4) and the symbol object.
    """
    symbols = []
    dst_pos = 0
    cursor = offset
    
    bytes_left = uncompressed_len
    _log.warn(f"{bytes_left=},{cursor=}")
    while bytes_left > 0:
        result = parse_symbol(comp_stream,cursor,bytes_left,dst_pos)
        if result == None:
            _log.warn(f"stopped at {cursor=}")
            break
        symbol_dst_pos = dst_pos
        symbol_cursor = cursor
        (cursor, bytes_left, dst_pos, symbol) = result
        symbol_num_bytes = cursor - symbol_cursor
        symbols.append((symbol_cursor,symbol_num_bytes,symbol_dst_pos,symbol))
    return symbols

class LZ77Symbol:
    
    def __init__(self,length,
                      copy_offset,
                      comp_stream,
                      comp_stream_cursor):
        self.length = length
        self.copy_offset = copy_offset
        # metadata
        self._comp_stream = comp_stream
        self._comp_stream_cursor = comp_stream_cursor #: nvcomp Snappy does not store this but reconstructs this from a warp's read position plus the lengths of previous symbols

    @property
    def is_literal(self):
        return self.copy_offset < 0
    
    @property
    def is_copy(self):
        return not self.is_literal

    def __repr__(self):
        if self.is_literal:
            begin = -self.copy_offset
            end = begin + self.length
            # self._comp_stream_cursor+1:self._comp_stream_cursor+1+self.length
            return f"(\'{''.join((f'{a:02X}' for a in self._comp_stream[begin:end]))}\',len={self.length})"
        else:
            return f"<{self.copy_offset},len={self.length}>"

    __str__ = __repr__

    def get_bytes(self,output_buffer):
        if self.is_literal:
            return self._comp_stream[self._comp_stream_cursor+1:self._comp_stream_cursor+1+self.length]
        else:
            copied_literal_start_idx = self._comp_stream_cursor - self.copy_offset
            if self.length <= self.copy_offset:
                return self._comp_stream[copied_literal_index:self.length]
            else:
                pass

def symbol_len(b0):
    pass    


def parse_symbol(comp_stream,cursor,bytes_left,dst_pos):
    orig_cursor = cursor 

    def READ_BYTE(idx):
        nonlocal comp_stream
        return comp_stream[idx]

    b0 = READ_BYTE(cursor)
    old_cursor = cursor
    if b0 & 3:
        b1 = READ_BYTE(cursor+1)
        if not (b0 & 2):
            _log.warn(f"{cursor=}: found copy with 3-bit length, 11-bit offset")
            # xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
            offset = ((b0 & 0xe0) << 3) | b1
            blen   = ((b0 >> 2) & 7) + 4
            cursor += 2
        else: 
            _log.warn(f"{cursor=}: found copy with 6-bit length, 2-byte or 4-byte offset")
            # xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
            offset = b1 | (READ_BYTE(cursor + 2) << 8)
            if (b0 & 1):  #  4-byte offset
                offset |= (READ_BYTE(cursor + 3) << 16) | (READ_BYTE(cursor + 4) << 24)
                cursor += 5
            else:
                cursor += 3
            blen = (b0 >> 2) + 1
        dst_pos += blen
        if (offset - 1 >= dst_pos or bytes_left < blen):
            cursor = old_cursor
            _log.warn(f"{cursor=}: out of range or not enough bytes left, {bytes_left=}, {blen=}")
            return None 
        bytes_left -= blen
    elif (b0 < 4 * 4):
        _log.warn(f"{cursor=}: found short literal")
        # 0000xx00: short literal
        blen   = (b0 >> 2) + 1
        offset = -(cursor + 1)
        cursor += 1 + blen
        dst_pos += blen
        if (bytes_left < blen):
            cursor = old_cursor
            _log.warn(f"{cursor=}: not enough bytes left to construct literal, {bytes_left=}, {blen=}")
            return None 
        bytes_left -= blen
    else:
        _log.warn(f"{cursor=}: found literal")
        # xxxxxx00: literal
        blen = b0 >> 2
        if (blen >= 60):
            num_bytes = blen - 59
            blen               = READ_BYTE(cursor + 1)
            if (num_bytes > 1):
                blen |= READ_BYTE(cursor + 2) << 8
                if (num_bytes > 2):
                    blen |= READ_BYTE(cursor + 3) << 16
                if (num_bytes > 3): 
                    blen |= READ_BYTE(cursor + 4) << 24
            cursor += num_bytes
        cursor += 1
        blen += 1
        offset = -cursor
        cursor += blen
        dst_pos += blen
        if (bytes_left < blen):
            cursor = old_cursor
            _log.warn(f"{cursor=}: not enough bytes left to construct literal, {bytes_left=}, {blen=}")
            return None 
        bytes_left -= blen
        
    return cursor, bytes_left, dst_pos, LZ77Symbol(
        blen,
        offset,
        comp_stream=comp_stream,
        comp_stream_cursor=orig_cursor
    )

def process_symbols(lz77symbols):
    result = bytearray([])
    

