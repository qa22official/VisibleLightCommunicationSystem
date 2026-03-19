#!/usr/bin/env python3
"""GF(2^11) Reed-Solomon 核心工具。

本模块实现基于 11bit 数据元的 RS 编码与仅擦除恢复解码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

M = 11
FIELD_SIZE = 1 << M
FIELD_MASK = FIELD_SIZE - 1
# 本原多项式：x^11 + x^2 + 1
PRIMITIVE_POLY = 0x805


@dataclass(frozen=True)
class RS11Config:
    n: int = 255
    k: int = 242
    fcr: int = 0
    generator: int = 2

    @property
    def nsym(self) -> int:
        return self.n - self.k


class GF11:
    def __init__(self, primitive: int = PRIMITIVE_POLY) -> None:
        self.exp = [0] * (2 * (FIELD_SIZE - 1))
        self.log = [0] * FIELD_SIZE
        x = 1
        for i in range(FIELD_SIZE - 1):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & FIELD_SIZE:
                x ^= primitive
            x &= FIELD_MASK
        for i in range(FIELD_SIZE - 1, 2 * (FIELD_SIZE - 1)):
            self.exp[i] = self.exp[i - (FIELD_SIZE - 1)]

    @staticmethod
    def add(a: int, b: int) -> int:
        return a ^ b

    @staticmethod
    def sub(a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF(2^11) 中发生除零")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % (FIELD_SIZE - 1)]

    def pow(self, a: int, power: int) -> int:
        if power == 0:
            return 1
        if a == 0:
            return 0
        return self.exp[(self.log[a] * power) % (FIELD_SIZE - 1)]


class RS11:
    def __init__(self, config: RS11Config) -> None:
        if config.n <= 0 or config.k <= 0 or config.k >= config.n:
            raise ValueError("RS 参数无效")
        if config.n >= FIELD_SIZE:
            raise ValueError(f"在 GF(2^11) 中，n 必须小于 {FIELD_SIZE}")

        self.cfg = config
        self.gf = GF11()
        self.gen = self._make_generator(config.nsym)

    def _poly_mul(self, p: Sequence[int], q: Sequence[int]) -> List[int]:
        out = [0] * (len(p) + len(q) - 1)
        for i, pv in enumerate(p):
            if pv == 0:
                continue
            for j, qv in enumerate(q):
                if qv == 0:
                    continue
                out[i + j] ^= self.gf.mul(pv, qv)
        return out

    def _poly_eval(self, poly: Sequence[int], x: int) -> int:
        y = 0
        for coeff in poly:
            y = self.gf.mul(y, x) ^ coeff
        return y

    def _make_generator(self, nsym: int) -> List[int]:
        g = [1]
        for i in range(nsym):
            root = self.gf.pow(self.cfg.generator, self.cfg.fcr + i)
            g = self._poly_mul(g, [1, root])
        return g

    def encode_block(self, data: Sequence[int]) -> List[int]:
        if len(data) != self.cfg.k:
            raise ValueError(f"encode_block 需要 {self.cfg.k} 个数据元")

        nsym = self.cfg.nsym
        reg = [0] * nsym
        for sym in data:
            if sym < 0 or sym > FIELD_MASK:
                raise ValueError("数据元超出 11bit 取值范围")

            fb = sym ^ reg[0]
            reg = reg[1:] + [0]
            if fb != 0:
                for j in range(nsym):
                    coef = self.gen[j + 1]
                    if coef:
                        reg[j] ^= self.gf.mul(fb, coef)
        return list(data) + reg

    def syndromes(self, codeword: Sequence[int]) -> List[int]:
        if len(codeword) != self.cfg.n:
            raise ValueError(f"syndromes 需要 {self.cfg.n} 个数据元")

        out: List[int] = []
        for i in range(self.cfg.nsym):
            x = self.gf.pow(self.cfg.generator, self.cfg.fcr + i)
            out.append(self._poly_eval(codeword, x))
        return out

    @staticmethod
    def _is_zero(values: Sequence[int]) -> bool:
        return all(v == 0 for v in values)

    def _solve_linear(self, matrix: List[List[int]], rhs: List[int]) -> List[int]:
        n = len(rhs)
        aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

        for col in range(n):
            pivot = None
            for r in range(col, n):
                if aug[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                raise ValueError("擦除方程组奇异，无法恢复")

            if pivot != col:
                aug[col], aug[pivot] = aug[pivot], aug[col]

            inv = self.gf.div(1, aug[col][col])
            for c in range(col, n + 1):
                aug[col][c] = self.gf.mul(aug[col][c], inv)

            for r in range(n):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor == 0:
                    continue
                for c in range(col, n + 1):
                    aug[r][c] ^= self.gf.mul(factor, aug[col][c])

        return [aug[i][n] for i in range(n)]

    def correct_erasures(self, codeword: Sequence[int], erasures: Sequence[int]) -> List[int]:
        if len(codeword) != self.cfg.n:
            raise ValueError(f"correct_erasures 需要 {self.cfg.n} 个数据元")

        cw = list(codeword)
        if not erasures:
            syn = self.syndromes(cw)
            if not self._is_zero(syn):
                raise ValueError("码字存在错误，但未提供擦除位置信息")
            return cw

        uniq = sorted(set(erasures))
        if any(pos < 0 or pos >= self.cfg.n for pos in uniq):
            raise ValueError("擦除位置超出范围")
        if len(uniq) > self.cfg.nsym:
            raise ValueError(
                f"擦除数量过多（{len(uniq)}），最多可恢复 {self.cfg.nsym} 个"
            )

        for pos in uniq:
            cw[pos] = 0

        syn = self.syndromes(cw)
        e = len(uniq)
        matrix = [[0] * e for _ in range(e)]
        rhs = [syn[row] for row in range(e)]

        for row in range(e):
            exp_row = self.cfg.fcr + row
            for col, pos in enumerate(uniq):
                power = (self.cfg.n - 1 - pos) * exp_row
                matrix[row][col] = self.gf.pow(self.cfg.generator, power)

        values = self._solve_linear(matrix, rhs)
        for col, pos in enumerate(uniq):
            cw[pos] = values[col]

        check = self.syndromes(cw)
        if not self._is_zero(check):
            raise ValueError("擦除恢复后 syndrome 校验失败")
        return cw


def bytes_to_11bit_symbols(data: bytes) -> tuple[list[int], int]:
    acc = 0
    acc_bits = 0
    symbols: List[int] = []

    for b in data:
        acc = (acc << 8) | b
        acc_bits += 8
        while acc_bits >= M:
            acc_bits -= M
            symbols.append((acc >> acc_bits) & FIELD_MASK)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    pad_bits = 0
    if acc_bits > 0:
        pad_bits = M - acc_bits
        symbols.append((acc << pad_bits) & FIELD_MASK)
    return symbols, pad_bits


def symbols11_to_bytes(symbols: Sequence[int], pad_bits: int) -> bytes:
    if pad_bits < 0 or pad_bits >= M:
        raise ValueError("pad_bits 必须在 [0, 10] 区间内")

    acc = 0
    acc_bits = 0
    out = bytearray()

    for sym in symbols:
        if sym < 0 or sym > FIELD_MASK:
            raise ValueError("数据元超出 11bit 取值范围")
        acc = (acc << M) | sym
        acc_bits += M
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    if pad_bits > 0:
        # 丢弃为凑满最后一个 11bit 数据元而附加的尾部填充位。
        total_bits = len(symbols) * M - pad_bits
        total_bytes = total_bits // 8
        return bytes(out[:total_bytes])
    return bytes(out)


def pack_11bit_symbols(symbols: Sequence[int]) -> bytes:
    """将 11bit 数据元紧凑打包为字节流。"""
    acc = 0
    acc_bits = 0
    out = bytearray()

    for sym in symbols:
        if sym < 0 or sym > FIELD_MASK:
            raise ValueError("数据元超出 11bit 取值范围")
        acc = (acc << M) | sym
        acc_bits += M
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)

    return bytes(out)


def unpack_11bit_symbols(data: bytes, symbol_count: int) -> List[int]:
    """从紧凑字节流中解包出指定数量的 11bit 数据元。"""
    if symbol_count < 0:
        raise ValueError("symbol_count 必须大于等于 0")

    acc = 0
    acc_bits = 0
    out: List[int] = []

    for b in data:
        acc = (acc << 8) | b
        acc_bits += 8
        while acc_bits >= M and len(out) < symbol_count:
            acc_bits -= M
            out.append((acc >> acc_bits) & FIELD_MASK)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    if len(out) != symbol_count:
        raise ValueError(
            f"位数不足，无法解包足够的数据元：得到 {len(out)}，期望 {symbol_count}"
        )

    return out
