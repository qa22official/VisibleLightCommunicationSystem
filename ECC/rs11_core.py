#!/usr/bin/env python3
"""GF(2^11) Reed-Solomon 核心工具。

本模块实现基于 11bit 数据元的 RS 编码、擦除恢复，
以及“擦除恢复后 1 个随机错误”的兜底修复。
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

    def _poly_add_asc(self, p: Sequence[int], q: Sequence[int]) -> List[int]:
        m = max(len(p), len(q))
        out = [0] * m
        for i in range(m):
            a = p[i] if i < len(p) else 0
            b = q[i] if i < len(q) else 0
            out[i] = a ^ b
        return self._poly_trim_asc(out)

    def _poly_mul_asc(self, p: Sequence[int], q: Sequence[int]) -> List[int]:
        out = [0] * (len(p) + len(q) - 1)
        for i, pv in enumerate(p):
            if pv == 0:
                continue
            for j, qv in enumerate(q):
                if qv == 0:
                    continue
                out[i + j] ^= self.gf.mul(pv, qv)
        return self._poly_trim_asc(out)

    @staticmethod
    def _poly_trim_asc(p: Sequence[int]) -> List[int]:
        out = list(p)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    def _poly_eval_asc(self, poly: Sequence[int], x: int) -> int:
        y = 0
        xp = 1
        for coeff in poly:
            if coeff:
                y ^= self.gf.mul(coeff, xp)
            xp = self.gf.mul(xp, x)
        return y

    def _poly_scale_shift_asc(self, poly: Sequence[int], scale: int, shift: int) -> List[int]:
        if scale == 0:
            return [0]
        out = [0] * shift
        for v in poly:
            out.append(self.gf.mul(v, scale) if v else 0)
        return self._poly_trim_asc(out)

    def _locator_from_positions(self, positions: Sequence[int]) -> List[int]:
        # 升幂系数：Λ(z)=Π(1-X_i z), X_i=α^(n-1-pos_i)
        loc = [1]
        for pos in positions:
            x = self.gf.pow(self.cfg.generator, self.cfg.n - 1 - pos)
            loc = self._poly_mul_asc(loc, [1, x])
        return self._poly_trim_asc(loc)

    def _forney_syndromes(self, synd: Sequence[int], erasures: Sequence[int]) -> List[int]:
        out = list(synd)
        for pos in erasures:
            x = self.gf.pow(self.cfg.generator, self.cfg.n - 1 - pos)
            for i in range(len(out) - 1):
                out[i] = self.gf.mul(out[i], x) ^ out[i + 1]
            out.pop()
            if not out:
                break
        return out

    def _bm_error_locator(self, synd: Sequence[int], max_deg: int) -> List[int]:
        # Berlekamp-Massey，返回升幂系数 locator。
        c = [1]
        b = [1]
        l = 0
        m = 1
        bb = 1

        for n in range(len(synd)):
            d = synd[n]
            for i in range(1, l + 1):
                if i < len(c) and c[i] and synd[n - i]:
                    d ^= self.gf.mul(c[i], synd[n - i])

            if d == 0:
                m += 1
                continue

            t = c[:]
            scale = self.gf.div(d, bb)
            delta = self._poly_scale_shift_asc(b, scale, m)
            c = self._poly_add_asc(c, delta)

            if 2 * l <= n:
                l = n + 1 - l
                b = t
                bb = d
                m = 1
            else:
                m += 1

        c = self._poly_trim_asc(c)
        if len(c) - 1 > max_deg:
            raise ValueError("错误定位多项式次数超过可纠范围")
        return c

    def _find_error_positions_from_locator(self, locator: Sequence[int]) -> List[int]:
        # 根搜索：Λ(1/X)=0，其中 X=α^(n-1-pos)
        deg = len(locator) - 1
        if deg <= 0:
            return []

        out: List[int] = []
        for pos in range(self.cfg.n):
            x = self.gf.pow(self.cfg.generator, self.cfg.n - 1 - pos)
            z = self.gf.div(1, x)
            if self._poly_eval_asc(locator, z) == 0:
                out.append(pos)

        if len(out) != deg:
            raise ValueError("错误位置搜索失败，根数量与定位多项式次数不一致")
        return sorted(out)

    def _solve_errata_values(self, synd: Sequence[int], positions: Sequence[int]) -> List[int]:
        e = len(positions)
        matrix = [[0] * e for _ in range(e)]
        rhs = [synd[row] for row in range(e)]

        for row in range(e):
            exp_row = self.cfg.fcr + row
            for col, pos in enumerate(positions):
                power = (self.cfg.n - 1 - pos) * exp_row
                matrix[row][col] = self.gf.pow(self.cfg.generator, power)

        return self._solve_linear(matrix, rhs)

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

    def try_correct_one_error(self, codeword: Sequence[int]) -> List[int] | None:
        """尝试修复恰好 1 个未知错误（不含擦除）。

        条件：syndrome 必须满足单错误模型。若不满足返回 None。
        """
        if len(codeword) != self.cfg.n:
            raise ValueError(f"try_correct_one_error 需要 {self.cfg.n} 个数据元")

        syn = self.syndromes(codeword)
        if self._is_zero(syn):
            return list(codeword)
        if self.cfg.nsym < 2:
            return None

        s0 = syn[0]
        s1 = syn[1]
        if s0 == 0:
            return None

        x = self.gf.div(s1, s0)
        if x == 0:
            return None

        # 单错误模型：S_i = s0 * x^i（fcr=0 时成立）。
        power = 1
        for sval in syn:
            if sval != self.gf.mul(s0, power):
                return None
            power = self.gf.mul(power, x)

        logx = self.gf.log[x]
        pos = (self.cfg.n - 1 - logx) % (FIELD_SIZE - 1)
        if pos >= self.cfg.n:
            return None

        out = list(codeword)
        out[pos] ^= s0
        if not self._is_zero(self.syndromes(out)):
            return None
        return out

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
        if self._is_zero(check):
            return cw

        # 低概率 OCR 误识别场景：擦除恢复后允许再尝试 1 个随机错误修复。
        one_err_fixed = self.try_correct_one_error(cw)
        if one_err_fixed is not None:
            return one_err_fixed

        raise ValueError("擦除恢复后 syndrome 校验失败")

    def _correct_errors_and_erasures_impl(
        self, codeword: Sequence[int], erasures: Sequence[int] | None = None
    ) -> tuple[List[int], dict]:
        """完整 RS errors-and-erasures 解码实现，并返回统计信息。"""
        if len(codeword) != self.cfg.n:
            raise ValueError(f"correct_errors_and_erasures 需要 {self.cfg.n} 个数据元")

        cw = list(codeword)
        erasures = [] if erasures is None else sorted(set(erasures))
        if any(pos < 0 or pos >= self.cfg.n for pos in erasures):
            raise ValueError("擦除位置超出范围")
        if len(erasures) > self.cfg.nsym:
            raise ValueError("擦除数量超过可恢复上限")

        synd = self.syndromes(cw)
        if self._is_zero(synd):
            stats = {
                "erasures": len(erasures),
                "unknown_errors": 0,
                "total_errata": 0,
                "budget_used": len(erasures),
                "budget_total": self.cfg.nsym,
                "changed": False,
            }
            return cw, stats

        if len(erasures) == self.cfg.nsym:
            # 全部能力都被擦除占满，只能接受零未知错误。
            raise ValueError("擦除数量已占满纠错能力且 syndrome 非零")

        erase_loc = self._locator_from_positions(erasures)
        fsynd = self._forney_syndromes(synd, erasures)
        max_unknown = (self.cfg.nsym - len(erasures)) // 2
        err_loc = self._bm_error_locator(fsynd, max_unknown)
        locator = self._poly_mul_asc(erase_loc, err_loc)
        total_positions = self._find_error_positions_from_locator(locator)

        magnitudes = self._solve_errata_values(synd, total_positions)
        for pos, mag in zip(total_positions, magnitudes):
            cw[pos] ^= mag

        check = self.syndromes(cw)
        if not self._is_zero(check):
            raise ValueError("errors-and-erasures 解码后 syndrome 仍非零")

        total_errata = len(total_positions)
        unknown_errors = max(0, total_errata - len(erasures))
        stats = {
            "erasures": len(erasures),
            "unknown_errors": unknown_errors,
            "total_errata": total_errata,
            "budget_used": len(erasures) + 2 * unknown_errors,
            "budget_total": self.cfg.nsym,
            "changed": True,
        }
        return cw, stats

    def correct_errors_and_erasures(
        self, codeword: Sequence[int], erasures: Sequence[int] | None = None
    ) -> List[int]:
        """完整 RS errors-and-erasures 解码。

        约束：2t + s <= nsym，其中 t 为未知错误数，s 为已知擦除数。
        """
        cw, _stats = self._correct_errors_and_erasures_impl(codeword, erasures)
        return cw

    def correct_errors_and_erasures_with_stats(
        self, codeword: Sequence[int], erasures: Sequence[int] | None = None
    ) -> tuple[List[int], dict]:
        """完整 RS errors-and-erasures 解码，并返回纠错统计。"""
        return self._correct_errors_and_erasures_impl(codeword, erasures)

    def correct_erasures_plus_one_unknown(
        self, codeword: Sequence[int], erasures: Sequence[int]
    ) -> List[int] | None:
        """在已知擦除基础上，再容忍 1 个未知错误。

        做法：遍历 1 个候选位置并把它当作额外擦除，
        复用纯擦除求解器。若任一候选可通过 syndrome 校验则返回。
        """
        if len(codeword) != self.cfg.n:
            raise ValueError(f"correct_erasures_plus_one_unknown 需要 {self.cfg.n} 个数据元")

        base = sorted(set(erasures))
        if any(pos < 0 or pos >= self.cfg.n for pos in base):
            raise ValueError("擦除位置超出范围")
        if len(base) + 1 > self.cfg.nsym:
            return None

        known = set(base)
        for pos in range(self.cfg.n):
            if pos in known:
                continue
            trial = base + [pos]
            try:
                return self.correct_erasures(codeword, trial)
            except Exception:
                continue
        return None


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
