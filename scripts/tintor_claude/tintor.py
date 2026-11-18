"""Tintorovi pokazatelji (Poslovna analiza) kao ciste funkcije nad stavkama izvjestaja.

Sve stope su omjeri (ne postoci). Ulazi su pandas Series ili skalari u istoj valuti.
Oznake prate knjigu: A aktiva, VK vlastiti kapital, TK tudji kapital, TI tekuca imovina,
KO kratkorocne obveze, DI/DA dugotrajna imovina, DO dugorocne obveze, ND neto dobit,
BD bruto dobit (= dobit prije kamata i poreza, str. 327), UP ukupni prihod.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _div(a, b):
    """Omjer koji vraca NaN za nulti ili nedostajuci nazivnik."""
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    out = a / b.where(b != 0)
    return out


def bruto_dobit(ebt, interest_exp):
    """BD = dobit prije oporezivanja + kamate (str. 327: BD - kamate = dobit D; D - porez = ND)."""
    return pd.Series(ebt, dtype=float) + pd.Series(interest_exp, dtype=float).fillna(0)


def r_vk(nd, vk):
    """Rentabilnost vlastitog kapitala R_VK = ND / VK (str. 1032). Nedefinirano za VK <= 0."""
    vk = pd.Series(vk, dtype=float)
    return _div(nd, vk.where(vk > 0))


def roi(bd, a):
    """Rentabilnost imovine RoI = BD / A (str. 1032), BD = bruto dobit."""
    a = pd.Series(a, dtype=float)
    return _div(bd, a.where(a > 0))


def scf(cfo, capex):
    """Slobodni novcani tok SCF = CFO - kapitalni izdaci (formula 5.2, str. 338-339)."""
    return pd.Series(cfo, dtype=float) - pd.Series(capex, dtype=float).fillna(0)


def fp(tk, vk):
    """Financijska poluga Fp = TK / VK (str. 1032). Nedefinirano za VK <= 0."""
    vk = pd.Series(vk, dtype=float)
    return _div(tk, vk.where(vk > 0))


def kl(ti, ko):
    """Likvidnost KL = TI / KO (str. 1032)."""
    ko = pd.Series(ko, dtype=float)
    return _div(ti, ko.where(ko > 0))


def p1(vk, di):
    """Pokrice 1. stupnja P1 = VK / DI (str. 1032; 7.61)."""
    di = pd.Series(di, dtype=float)
    return _div(vk, di.where(di > 0))


def p2(vk, do, da):
    """Pokrice 2. stupnja P2 = (VK + DO) / DA (formula 7.62, str. 554-555)."""
    da = pd.Series(da, dtype=float)
    return _div(pd.Series(vk, dtype=float) + pd.Series(do, dtype=float).fillna(0), da.where(da > 0))


def altman_z(nrk, zd, bd, vk, tk, up, a):
    """Altmanov Z prema formuli 6.1 (str. 394):
    Z = 1,2 NRK/A + 1,4 ZD/A + 3,3 BD/A + 0,6 VK/TK + 1,0 UP/A.
    Knjizni primjer (tablica 6.2, str. 399) u drugom clanu koristi ND/A; za njega proslijedi zd=ND."""
    a = pd.Series(a, dtype=float)
    a = a.where(a > 0)
    tk = pd.Series(tk, dtype=float)
    return (1.2 * _div(nrk, a) + 1.4 * _div(zd, a) + 3.3 * _div(bd, a)
            + 0.6 * _div(vk, tk.where(tk > 0)) + 1.0 * _div(up, a))


def kralicek_df(cisti_nt, obveze, a, bd, pp, zalihe):
    """Kralicekov DF (formula 6.2, str. 395):
    DF = 1,5 x1 + 0,08 x2 + 10 x3 + 5 x4 + 0,3 x5 + 0,1 x6;
    x1 cisti novcani tok/obveze, x2 ukupna sredstva/obveze, x3 BD/ukupna sredstva,
    x4 BD/poslovni prihodi, x5 zalihe/poslovni prihodi, x6 poslovni prihodi/ukupna sredstva."""
    ob = pd.Series(obveze, dtype=float)
    ob = ob.where(ob > 0)
    a = pd.Series(a, dtype=float)
    a = a.where(a > 0)
    pp = pd.Series(pp, dtype=float)
    pp = pp.where(pp > 0)
    return (1.5 * _div(cisti_nt, ob) + 0.08 * _div(a, ob) + 10 * _div(bd, a) + 5 * _div(bd, pp)
            + 0.3 * _div(pd.Series(zalihe, dtype=float).fillna(0), pp) + 0.1 * _div(pp, a))


def self_test() -> None:
    """Provjera prema tablicama 6.1-6.3 (OSAP d.d., str. 399) i primjeru SCF-a (str. 338-339)."""
    t = pd.DataFrame({
        "nrk": [-380.82, 134.79, 230.45, 494.99], "a": [5335.30, 5084.88, 5186.40, 5101.68],
        "nd": [49.78, -61.33, 101.85, 128.25], "bd": [289.37, 169.36, 304.81, 258.85],
        "vk": [3382.47, 2970.24, 3072.06, 3200.34], "tk": [1952.83, 2108.88, 2108.59, 1895.59],
        "up": [3914.40, 3429.65, 3833.41, 4648.30]})
    z = altman_z(t.nrk, t.nd, t.bd, t.vk, t.tk, t.up, t.a).round(2).tolist()
    assert z == [1.88, 1.64, 1.89, 2.24], z  # tablica 6.3, redak 2
    assert float(scf(155, 45 - 30).iloc[0]) == 140.0  # 155 - 45 + 30
    assert np.isnan(r_vk(-10, -5).iloc[0]) and np.isnan(fp(100, -5).iloc[0])  # negativan kapital nije "visok" omjer
    assert round(float(fp(1952.83, 3382.47).iloc[0]), 2) == 0.58


if __name__ == "__main__":
    self_test()
    print("tintor.py: knjizni primjeri reproducirani (Altman Z 1,88 / 1,64 / 1,89 / 2,24; SCF 140)")
