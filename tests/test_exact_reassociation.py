"""Independent exhaustive algebra checks; no CUDA or miner dependency."""
import random
import re
import struct
from pathlib import Path


def main():
    table = Path(__file__).parents[1] / 'src/cuda/cp_radix_lut.h'
    rows = [list(map(int, m.split(','))) for m in re.findall(r'\{([-\d,]+)\}', table.read_text())]
    assert len(rows) == 129
    allowed = {-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6}
    for x, d in zip(range(-64, 65), rows):
        assert set(d) <= allowed
        assert d[0]+4*d[1]+16*d[2] == x
        assert d[3:] == [d[0]+d[1], d[0]+d[2], d[1]+d[2]]
        for y, e in zip(range(-64, 65), rows):
            assert sum(w*a*b for w, a, b in zip((-19, -52, 176, 4, 16, 64), d, e)) == x*y
            h, l = (x+4)//8, x-8*((x+4)//8)
            hh, ll = (y+4)//8, y-8*((y+4)//8)
            assert 56*h*hh-7*l*ll+8*(h+l)*(hh+ll) == x*y
            # Half-scaled FP6 products use ULP=1/4, recovering the same integer coefficients.
            assert 4*(56*(h/2)*(hh/2)-7*(l/2)*(ll/2)+8*((h+l)/2)*((hh+ll)/2)) == x*y
            hi, lo = (x+8)//16, x-16*((x+8)//16)
            hj, lj = (y+8)//16, y-16*((y+8)//16)
            assert 256*hi*hj+16*(hi*lj+lo*hj)+lo*lj == x*y
    rng = random.Random(8123)
    pairs = [(rng.randrange(128), rng.randrange(128)) for _ in range(4096)]
    a = [rng.randrange(-63, 65) for _ in pairs]
    e = [rng.randrange(-16, 48) for _ in range(128)]
    prefix = [0]*128
    c = 0
    for k, (p, q) in enumerate(pairs):
        prefix[p] += a[k]
        prefix[q] -= a[k]
        c += a[k]*(e[p]-e[q])
        if (k+1) % 128 == 0:
            assert sum(x*y for x, y in zip(prefix, e)) == c
    # Check the bias/XOR recovery across both signs and binade boundaries.
    values = [-4194303, -1, 0, 1, 4194303]
    values += [rng.randrange(-4194303, 4194304) for _ in range(59)]
    encoded = expected = 0
    for x in values:
        encoded ^= struct.unpack('<I', struct.pack('<f', 12582912.0+x))[0]
        expected ^= x & 0xffffffff
    signed = ((encoded & 0x7fffff) ^ 0x400000)-0x400000
    assert signed & 0xffffffff == expected
    print('PASS: 16641 integer pairs per radix identity, 32 projection milestones, bias/XOR recovery')


if __name__ == '__main__':
    main()
