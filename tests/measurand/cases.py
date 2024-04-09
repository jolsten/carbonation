from dataclasses import dataclass


@dataclass
class Example:
    spec: str
    result: int
    word_size: int = 8
    one_based: bool = True
    size: int = 8


component_test_cases = [
    Example("1", 1),
    Example("0", 1, one_based=False),
    Example("128", 128),
    Example("255", 255),
    Example("1:1", 1, size=1),
    Example("256:1", 0, size=1),
    Example("170:1-4", 0xA, size=4),
    Example("170:5-8", 0xA, size=4),
    Example("1R", 128),
    Example("128R", 1),
    Example("170:1-4R", 0x5, size=4),
    Example("170:5-8R", 0x5, size=4),
    Example("170", 0x0AA, word_size=12, size=12),
    Example("170R", 0x550, word_size=12, size=12),
    Example("2730", 0xAAA, word_size=12, size=12),
    Example("2730R", 0x555, word_size=12, size=12),
    Example("4095", 0xFFF, word_size=12, size=12),
    Example("4095:1-4", 0x00F, word_size=12, size=4),
    Example("4095:5-8", 0x00F, word_size=12, size=4),
    Example("4095:9-12", 0x00F, word_size=12, size=4),
    Example("4095:1-8", 0x0FF, word_size=12, size=8),
    Example("4095:5-12", 0x0FF, word_size=12, size=8),
]

parameter_test_cases = [
    Example("1+2", 0x0102, size=16),
    Example("255+255", 0xFFFF, size=16),
    Example("255+255+255", 0xFFFFFF, size=24),
    Example("255+255+255+255", 0xFFFFFFFF, size=32),
    Example("255+255+255+255+255+255+255+255", 0xFFFFFFFFFFFFFFFF, size=64),
    Example("1+256", 0x0100, size=16),
    Example("256+1", 0x0001, size=16),
    Example("1", 0x01, word_size=8, size=8),
    Example("2", 0x02, word_size=10, size=10),
    Example("1+2", 0x0102, word_size=8, size=16),
    Example("1:1-4+2:5-8", 0b0001_0000, word_size=8, size=8),
    Example("1-4", 0x01020304, word_size=8, size=32),
    Example("1-3", 0b0000000001_0000000010_0000000011, word_size=10, size=30),
] + component_test_cases

measurand_test_cases = (
    [
        Example("1;2c", result=1, word_size=8, size=8),
        Example("255;2c", result=-1, word_size=8, size=8),
        # MIL-STD-1750A32
        Example("127+255+255+127;1750a32", result=0.9999998 * 2**127),
        # (0x7FFFFF7F, 0.9999998 * 2**127),
        Example("64+256+256+127;1750a32", result=0.5 * 2**127),
        # (0x4000007F, 0.5 * 2**127),
        Example("80+256+256+4;1750a32", result=0.625 * 2**4),
        # (0x50000004, 0.625 * 2**4),
        Example("64+256+256+1;1750a32", result=0.5 * 2**1),
        # (0x40000001, 0.5 * 2**1),
        Example("64+256+256+256;1750a32", result=0.5 * 2**0),
        # (0x40000000, 0.5 * 2**0),
        Example("64+256+256+255;1750a32", result=0.5 * 2**-1),
        # (0x400000FF, 0.5 * 2**-1),
        Example("64+256+256+128;1750a32", result=0.5 * 2**-128),
        # (0x40000080, 0.5 * 2**-128),
        Example("256+256+256+256;1750a32", result=0.0 * 2**0),
        # (0x00000000, 0.0 * 2**0),
        # (0x80000000, -1.0 * 2**0),
        # (0xBFFFFF80, -0.5000001 * 2**-128),
        # (0x9FFFFF04, -0.7500001 * 2**4),
    ]
    + parameter_test_cases
    + component_test_cases
)
