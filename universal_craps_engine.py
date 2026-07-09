#!/usr/bin/env python3
"""A rules-first, YAML-driven, object-oriented craps simulation engine.

Created by Greg M. Krsak (greg.krsak@gmail.com) and the GPT-5.5 High setting
in ChatGPT Plus.

Licensed under the MIT open-source license. A copy of the license is available
in the LICENSE file, or via the web from: https://opensource.org/license/mit

The engine is kept in one file so it is easy to audit, copy, archive, and run
without installing a package.  Its internal design follows a Model-View-
Controller architecture:

* MODEL
    Immutable dice rolls, table rules, wagers, settlements, a cash ledger, and
    the complete state of the table live here.  The model knows what the game
    *is*, but it does not decide when a player should make a wager.

* VIEW
    Views translate model events into human-readable output.  A NullView makes
    Monte Carlo runs silent; ConsoleView produces an audit trail suitable for
    debugging a strategy one roll at a time.

* CONTROLLER
    The controller is the only layer allowed to change table state.  It accepts
    wager instructions, requests dice, resolves every active wager against an
    immutable pre-roll snapshot, advances the puck, and emits events.  A YAML
    strategy controller subscribes to those events and performs declarative
    actions.

Money is represented with Decimal rather than float.  A wager is removed from
cash when it is placed and remains part of equity while active.  A persistent
wager such as a Place 6 receives only its profit after a win because its stake
remains on the layout.  A one-roll wager receives its surviving stake and its
profit, then leaves the layout.  These accounting rules keep the ledger
consistent and prevent duplicate, missing, or incorrectly returned funds.

The implementation covers the principal casino-craps wager families:

* Pass Line, Don't Pass, Come, and Don't Come
* Taking and laying odds, including configurable 3-4-5x limits
* Place, Buy, Lay, Big 6, Big 8, and Hardway wagers
* Field, Any Seven, Any Craps, Aces, Ace-Deuce, Yo, and Boxcars
* Horn, C & E, World/Whirl, and exact unordered Hop combinations

Casino practices differ.  The companion YAML file therefore exposes field
payouts, the Don't bar number, vig timing and basis, odds limits, working-on-
come-out defaults, chip rounding, minimums, and legal wager increments.

Run modes:

    python universal_craps_engine.py strategy.yaml --simulate
    python universal_craps_engine.py strategy.yaml --run-scenarios
    python universal_craps_engine.py strategy.yaml --audit
    python universal_craps_engine.py --run-tests

The embedded unittest suite ships with the engine.  It uses deterministic
rolls, exhaustive 36-combination checks, ledger invariants, YAML validation,
and end-to-end scenario tests.
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys
import unittest
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence
from unittest import mock

import yaml


# ---------------------------------------------------------------------------
# MODEL: SMALL, PURE VALUE TYPES
# ---------------------------------------------------------------------------


Money = Decimal
POINT_NUMBERS = (4, 5, 6, 8, 9, 10)
CRAPS_NUMBERS = (2, 3, 12)
NATURALS = (7, 11)


def as_money(value: Any) -> Money:
    """Convert a YAML/Python numeric value to an exact Decimal amount."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def money_text(value: Money) -> str:
    """Return a stable, pleasant dollar representation for reports."""
    return f"${value.quantize(Decimal('0.01')):,.2f}"


class CrapsError(Exception):
    """Base exception for all intentional engine failures."""


class ConfigurationError(CrapsError):
    """Raised when YAML is unknown, contradictory, or incomplete."""


class BetError(CrapsError):
    """Raised when a wager is illegal in the current table state."""


class InsufficientFunds(BetError):
    """Raised when cash cannot cover a stake, increase, or commission."""


class Phase(str, Enum):
    """The puck has only two meaningful states in traditional craps."""

    COME_OUT = "come_out"
    POINT_ON = "point_on"


class BetKind(str, Enum):
    """Every wager family understood by the settlement engine."""

    PASS_LINE = "pass_line"
    DONT_PASS = "dont_pass"
    COME = "come"
    DONT_COME = "dont_come"
    TAKE_ODDS = "take_odds"
    LAY_ODDS = "lay_odds"
    PLACE = "place"
    BUY = "buy"
    LAY = "lay"
    FIELD = "field"
    BIG_6 = "big_6"
    BIG_8 = "big_8"
    HARDWAY = "hardway"
    ANY_SEVEN = "any_seven"
    ANY_CRAPS = "any_craps"
    ACES = "aces"
    ACE_DEUCE = "ace_deuce"
    YO = "yo"
    BOXCARS = "boxcars"
    HORN = "horn"
    C_AND_E = "c_and_e"
    WORLD = "world"
    HOP = "hop"


class BetStatus(str, Enum):
    """Lifecycle states for a wager stored in the table model."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    REMOVED = "removed"


class Outcome(str, Enum):
    """A settlement vocabulary designed for both people and tests."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    TRAVEL = "travel"
    NO_ACTION = "no_action"
    RETURNED_OFF = "returned_off"
    REMOVED = "removed"


@dataclass(frozen=True)
class DiceRoll:
    """An immutable pair of fair-die faces with useful derived properties."""

    die_1: int
    die_2: int

    def __post_init__(self) -> None:
        """Validate both die faces after dataclass initialization."""
        for face in (self.die_1, self.die_2):
            if face not in range(1, 7):
                raise ValueError("Each die face must be between 1 and 6.")

    @property
    def total(self) -> int:
        """Return the sum called by the stickperson."""
        return self.die_1 + self.die_2

    @property
    def is_pair(self) -> bool:
        """Return True for doubles, the defining feature of a hardway."""
        return self.die_1 == self.die_2

    @property
    def unordered_pair(self) -> tuple[int, int]:
        """Normalize a Hop combination so 3-4 and 4-3 are identical."""
        return tuple(sorted((self.die_1, self.die_2)))


class DiceSource(ABC):
    """Dependency-injection seam for random and deterministic dice."""

    @abstractmethod
    def roll(self) -> DiceRoll:
        """Return the next two-die result."""


class RandomDice(DiceSource):
    """Two independent discrete uniform draws from one seeded generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Create a private random generator with an optional seed."""
        self._random = random.Random(seed)

    def roll(self) -> DiceRoll:
        """Return one independent two-die result from the generator."""
        return DiceRoll(
            self._random.randint(1, 6),
            self._random.randint(1, 6),
        )


class ScriptedDice(DiceSource):
    """A finite deterministic source used by audits and unit tests."""

    def __init__(self, rolls: Iterable[Sequence[int]]) -> None:
        """Store an iterator over a finite sequence of predetermined rolls."""
        self._rolls: Iterator[Sequence[int]] = iter(rolls)

    def roll(self) -> DiceRoll:
        """Return the next scripted roll.

        Raise CrapsError when the configured sequence is exhausted.
        """
        try:
            pair = next(self._rolls)
        except StopIteration as exc:
            raise CrapsError(
                "The scripted dice sequence is exhausted."
            ) from exc
        if len(pair) != 2:
            raise ValueError("Every scripted roll must contain two die faces.")
        return DiceRoll(int(pair[0]), int(pair[1]))


@dataclass(frozen=True)
class OddsSchedule:
    """Maximum odds stakes, expressed as multiples of the flat wager.

    A familiar 3-4-5x table permits a Pass/Come odds stake of three times the
    flat wager on 4/10, four times on 5/9, and five times on 6/8.  The matching
    Don't schedule usually permits a six-times lay on every point; those lays
    win three, four, or five times the flat wager respectively.
    """

    take: Mapping[int, Money]
    lay: Mapping[int, Money]

    @classmethod
    def standard_345x(cls) -> "OddsSchedule":
        """Build the conventional 3-4-5x take and six-times lay schedule."""
        return cls(
            take={
                4: Decimal("3"),
                5: Decimal("4"),
                6: Decimal("5"),
                8: Decimal("5"),
                9: Decimal("4"),
                10: Decimal("3"),
            },
            lay={number: Decimal("6") for number in POINT_NUMBERS},
        )


@dataclass(frozen=True)
class TableRules:
    """Casino-specific rules that must never be hidden in settlement code."""

    dont_bar: int = 12
    field_2_profit: Money = Decimal("2")
    field_12_profit: Money = Decimal("3")
    buy_vig_rate: Money = Decimal("0.05")
    buy_vig_timing: str = "on_win"
    buy_vig_basis: str = "wager"
    lay_vig_rate: Money = Decimal("0.05")
    lay_vig_timing: str = "on_win"
    lay_vig_basis: str = "win"
    chip_unit: Money = Decimal("0.01")
    payout_rounding: str = "floor"
    enforce_increments: bool = True
    place_working_on_comeout: bool = False
    buy_working_on_comeout: bool = False
    lay_working_on_comeout: bool = False
    hardways_working_on_comeout: bool = False
    big_6_8_working_on_comeout: bool = False
    come_odds_working_on_comeout: bool = False
    dont_come_odds_working_on_comeout: bool = True
    odds: OddsSchedule = field(default_factory=OddsSchedule.standard_345x)
    minimums: Mapping[str, Any] = field(default_factory=dict)
    increments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate rule values that have a fixed set of legal choices."""
        if self.dont_bar not in (2, 12):
            raise ConfigurationError("dont_bar must be either 2 or 12.")
        for timing in (self.buy_vig_timing, self.lay_vig_timing):
            if timing not in ("upfront", "on_win"):
                raise ConfigurationError(
                    "Vig timing must be 'upfront' or 'on_win'."
                )
        for basis in (self.buy_vig_basis, self.lay_vig_basis):
            if basis not in ("wager", "win"):
                raise ConfigurationError(
                    "Vig basis must be 'wager' or 'win'."
                )
        if self.payout_rounding not in ("floor", "nearest", "exact"):
            raise ConfigurationError(
                "payout_rounding must be floor, nearest, or exact."
            )
        if self.chip_unit <= 0:
            raise ConfigurationError("chip_unit must be positive.")

    def round_money(self, value: Money) -> Money:
        """Apply the configured chip policy to a profit or commission."""
        if self.payout_rounding == "exact":
            return value
        units = value / self.chip_unit
        rounding = (
            ROUND_DOWN
            if self.payout_rounding == "floor"
            else ROUND_HALF_UP
        )
        return units.quantize(Decimal("1"), rounding=rounding) * self.chip_unit

    def minimum_for(self, kind: BetKind, number: Optional[int]) -> Money:
        """Return the configured minimum for a wager and optional number."""
        return self._numbered_rule(self.minimums, kind, number, Decimal("1"))

    def increment_for(self, kind: BetKind, number: Optional[int]) -> Money:
        """Return the legal chip increment for a wager and optional number."""
        return self._numbered_rule(
            self.increments,
            kind,
            number,
            self.chip_unit,
        )

    @staticmethod
    def _numbered_rule(
        rules: Mapping[str, Any],
        kind: BetKind,
        number: Optional[int],
        default: Money,
    ) -> Money:
        """Resolve a scalar or point-specific minimum or increment rule."""
        raw = rules.get(kind.value, default)
        if isinstance(raw, Mapping):
            if number is None:
                raw = raw.get("default", default)
            else:
                raw = raw.get(number, raw.get(str(number), default))
        return as_money(raw)


@dataclass
class Bet:
    """One reserved stake and all information needed to settle it."""

    bet_id: int
    kind: BetKind
    amount: Money
    number: Optional[int] = None
    parent_id: Optional[int] = None
    hop_pair: Optional[tuple[int, int]] = None
    working: Optional[bool] = None
    status: BetStatus = BetStatus.ACTIVE
    created_roll: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Report whether the wager remains available for settlement.

        Active wagers may also be selected by strategy actions.
        """
        return self.status == BetStatus.ACTIVE

    @property
    def is_persistent(self) -> bool:
        """Return True when a win normally leaves the stake on the layout."""
        return self.kind in {
            BetKind.PLACE,
            BetKind.BUY,
            BetKind.LAY,
            BetKind.BIG_6,
            BetKind.BIG_8,
            BetKind.HARDWAY,
        }


@dataclass(frozen=True)
class BetSnapshot:
    """Immutable copy used to make resolution independent of loop order."""

    bet_id: int
    kind: BetKind
    amount: Money
    number: Optional[int]
    parent_id: Optional[int]
    hop_pair: Optional[tuple[int, int]]
    working: Optional[bool]
    status: BetStatus

    @classmethod
    def from_bet(cls, bet: Bet) -> "BetSnapshot":
        """Create an immutable settlement snapshot from a mutable wager."""
        return cls(
            bet_id=bet.bet_id,
            kind=bet.kind,
            amount=bet.amount,
            number=bet.number,
            parent_id=bet.parent_id,
            hop_pair=bet.hop_pair,
            working=bet.working,
            status=bet.status,
        )


@dataclass(frozen=True)
class TableSnapshot:
    """The entire game state as it existed before the dice landed."""

    phase: Phase
    point: Optional[int]
    roll_number: int
    hand_number: int
    bets: Mapping[int, BetSnapshot]


@dataclass(frozen=True)
class Settlement:
    """A complete, auditable instruction for one wager after one roll."""

    bet_id: int
    outcome: Outcome
    credit: Money = Decimal("0")
    profit: Money = Decimal("0")
    commission: Money = Decimal("0")
    keep_active: bool = True
    travel_to: Optional[int] = None
    note: str = ""


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable cash movement in chronological order."""

    roll_number: int
    amount: Money
    balance: Money
    memo: str


class BankrollLedger:
    """Cash accounting with no implicit or reversible side effects."""

    def __init__(self, opening_cash: Money) -> None:
        """Initialize the ledger with nonnegative opening cash."""
        if opening_cash < 0:
            raise ValueError("Opening cash cannot be negative.")
        self.opening_cash = opening_cash
        self.cash = opening_cash
        self.entries: list[LedgerEntry] = []

    def debit(self, amount: Money, roll_number: int, memo: str) -> None:
        """Reserve cash and append an auditable negative ledger entry."""
        if amount < 0:
            raise ValueError("A debit amount cannot be negative.")
        if amount > self.cash:
            raise InsufficientFunds(
                f"Need {money_text(amount)} but only "
                f"{money_text(self.cash)} is available."
            )
        self.cash -= amount
        self.entries.append(
            LedgerEntry(roll_number, -amount, self.cash, memo)
        )

    def credit(self, amount: Money, roll_number: int, memo: str) -> None:
        """Return cash and append an auditable positive ledger entry."""
        if amount < 0:
            raise ValueError("A credit amount cannot be negative.")
        self.cash += amount
        self.entries.append(
            LedgerEntry(roll_number, amount, self.cash, memo)
        )


@dataclass
class TableModel:
    """The complete mutable model; only CrapsController may mutate it."""

    ledger: BankrollLedger
    rules: TableRules
    point: Optional[int] = None
    hand_number: int = 1
    roll_number: int = 0
    hand_roll_number: int = 0
    bets: dict[int, Bet] = field(default_factory=dict)
    next_bet_id: int = 1
    session_stopped: bool = False
    stop_reason: str = ""

    @property
    def phase(self) -> Phase:
        """Derive the table phase from whether a point is established."""
        return Phase.COME_OUT if self.point is None else Phase.POINT_ON

    @property
    def active_stake(self) -> Money:
        """Return the total stake currently reserved on active wagers."""
        return sum(
            (bet.amount for bet in self.bets.values() if bet.is_active),
            Decimal("0"),
        )

    @property
    def equity(self) -> Money:
        """Return rack cash plus every active wagered stake."""
        return self.ledger.cash + self.active_stake

    def active_bets(
        self,
        kind: Optional[BetKind] = None,
        number: Optional[int] = None,
        parent_id: Optional[int] = None,
    ) -> list[Bet]:
        """Filter active wagers by kind, number, and parent contract."""
        result = [bet for bet in self.bets.values() if bet.is_active]
        if kind is not None:
            result = [bet for bet in result if bet.kind == kind]
        if number is not None:
            result = [bet for bet in result if bet.number == number]
        if parent_id is not None:
            result = [bet for bet in result if bet.parent_id == parent_id]
        return result

    def snapshot(self) -> TableSnapshot:
        """Freeze the pre-roll table state used by the settlement engine."""
        return TableSnapshot(
            phase=self.phase,
            point=self.point,
            roll_number=self.roll_number,
            hand_number=self.hand_number,
            bets={
                bet_id: BetSnapshot.from_bet(bet)
                for bet_id, bet in self.bets.items()
                if bet.is_active
            },
        )


@dataclass(frozen=True)
class RollEvent:
    """The public record produced by a single controller roll."""

    roll: DiceRoll
    phase_before: Phase
    point_before: Optional[int]
    phase_after: Phase
    point_after: Optional[int]
    settlements: tuple[Settlement, ...]
    point_established: bool = False
    point_made: bool = False
    seven_out: bool = False


@dataclass(frozen=True)
class SessionResult:
    """A compact result suitable for reports and Monte Carlo aggregation."""

    opening_cash: Money
    closing_cash: Money
    closing_equity: Money
    rolls: int
    hands: int
    target_reached: bool
    stop_reason: str

    @property
    def net(self) -> Money:
        """Return the session gain or loss measured by closing equity."""
        return self.closing_equity - self.opening_cash


# ---------------------------------------------------------------------------
# MODEL: PAYOUT MATH AND PURE SETTLEMENT RULES
# ---------------------------------------------------------------------------


class PayoutMath:
    """Central home for payout ratios, splits, and commission arithmetic."""

    PLACE_PROFIT = {
        4: Decimal("1.8"),
        5: Decimal("1.4"),
        6: Decimal(7) / Decimal(6),
        8: Decimal(7) / Decimal(6),
        9: Decimal("1.4"),
        10: Decimal("1.8"),
    }
    TRUE_WIN_PROFIT = {
        4: Decimal("2"),
        5: Decimal("1.5"),
        6: Decimal("1.2"),
        8: Decimal("1.2"),
        9: Decimal("1.5"),
        10: Decimal("2"),
    }
    TRUE_LAY_PROFIT = {
        4: Decimal("0.5"),
        5: Decimal(2) / Decimal(3),
        6: Decimal(5) / Decimal(6),
        8: Decimal(5) / Decimal(6),
        9: Decimal(2) / Decimal(3),
        10: Decimal("0.5"),
    }
    HARDWAY_PROFIT = {
        4: Decimal("7"),
        6: Decimal("9"),
        8: Decimal("9"),
        10: Decimal("7"),
    }
    PROPOSITION_PROFIT = {
        BetKind.ANY_SEVEN: Decimal("4"),
        BetKind.ANY_CRAPS: Decimal("7"),
        BetKind.ACES: Decimal("30"),
        BetKind.ACE_DEUCE: Decimal("15"),
        BetKind.YO: Decimal("15"),
        BetKind.BOXCARS: Decimal("30"),
    }

    @staticmethod
    def gross_profit(
        amount: Money,
        ratio: Money,
        rules: TableRules,
    ) -> Money:
        """Calculate and round gross profit under the table chip policy."""
        return rules.round_money(amount * ratio)

    @staticmethod
    def vig(
        amount: Money,
        gross_win: Money,
        rate: Money,
        basis: str,
        rules: TableRules,
    ) -> Money:
        """Calculate commission from the configured basis and rate.

        Apply the table's chip-rounding policy before returning the result.
        """
        base = amount if basis == "wager" else gross_win
        return rules.round_money(base * rate)


class BetResolver:
    """Pure rules engine: snapshot plus roll in, settlement instruction out."""

    @classmethod
    def resolve_all(
        cls,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> list[Settlement]:
        """Resolve every active wager against the same pre-roll snapshot.

        Each settlement is calculated independently from loop order.
        """
        return [
            cls.resolve(bet, snapshot, roll, rules)
            for bet in snapshot.bets.values()
        ]

    @classmethod
    def resolve(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Dispatch a wager snapshot to its wager-specific settlement rule."""
        dispatch = {
            BetKind.PASS_LINE: cls._pass_line,
            BetKind.DONT_PASS: cls._dont_pass,
            BetKind.COME: cls._come,
            BetKind.DONT_COME: cls._dont_come,
            BetKind.TAKE_ODDS: cls._take_odds,
            BetKind.LAY_ODDS: cls._lay_odds,
            BetKind.PLACE: cls._place,
            BetKind.BUY: cls._buy,
            BetKind.LAY: cls._lay,
            BetKind.FIELD: cls._field,
            BetKind.BIG_6: cls._big_6,
            BetKind.BIG_8: cls._big_8,
            BetKind.HARDWAY: cls._hardway,
            BetKind.ANY_SEVEN: cls._proposition,
            BetKind.ANY_CRAPS: cls._proposition,
            BetKind.ACES: cls._proposition,
            BetKind.ACE_DEUCE: cls._proposition,
            BetKind.YO: cls._proposition,
            BetKind.BOXCARS: cls._proposition,
            BetKind.HORN: cls._horn,
            BetKind.C_AND_E: cls._c_and_e,
            BetKind.WORLD: cls._world,
            BetKind.HOP: cls._hop,
        }
        return dispatch[bet.kind](bet, snapshot, roll, rules)

    @staticmethod
    def _no_action(bet: BetSnapshot, note: str = "") -> Settlement:
        """Create a settlement that leaves the wager unchanged."""
        return Settlement(bet.bet_id, Outcome.NO_ACTION, note=note)

    @staticmethod
    def _loss(bet: BetSnapshot, note: str = "") -> Settlement:
        """Create a losing settlement that removes the wagered stake."""
        return Settlement(
            bet.bet_id,
            Outcome.LOSS,
            keep_active=False,
            note=note,
        )

    @staticmethod
    def _push(bet: BetSnapshot, note: str = "") -> Settlement:
        """Create a push settlement that leaves the contract active."""
        return Settlement(bet.bet_id, Outcome.PUSH, note=note)

    @staticmethod
    def _travel(
        bet: BetSnapshot,
        number: int,
        note: str,
    ) -> Settlement:
        """Move a Come-family contract from its box to a number."""
        return Settlement(
            bet.bet_id,
            Outcome.TRAVEL,
            travel_to=number,
            note=note,
        )

    @staticmethod
    def _contract_win(
        bet: BetSnapshot,
        profit: Money,
        note: str,
        commission: Money = Decimal("0"),
    ) -> Settlement:
        """Settle a winning contract or one-roll wager.

        Return its stake and net profit, then remove it from the layout.
        """
        net_profit = profit - commission
        return Settlement(
            bet_id=bet.bet_id,
            outcome=Outcome.WIN,
            credit=bet.amount + net_profit,
            profit=net_profit,
            commission=commission,
            keep_active=False,
            note=note,
        )

    @staticmethod
    def _persistent_win(
        bet: BetSnapshot,
        profit: Money,
        note: str,
        commission: Money = Decimal("0"),
    ) -> Settlement:
        """Credit profit while leaving a persistent wager on the layout."""
        net_profit = profit - commission
        return Settlement(
            bet_id=bet.bet_id,
            outcome=Outcome.WIN,
            credit=net_profit,
            profit=net_profit,
            commission=commission,
            keep_active=True,
            note=note,
        )

    @classmethod
    def _pass_line(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve Pass Line behavior for come-out and point-on phases."""
        del rules
        if snapshot.phase == Phase.COME_OUT:
            if roll.total in NATURALS:
                return cls._contract_win(
                    bet,
                    bet.amount,
                    "Pass Line wins on the come-out natural.",
                )
            if roll.total in CRAPS_NUMBERS:
                return cls._loss(
                    bet,
                    "Pass Line loses on come-out craps.",
                )
            return cls._no_action(
                bet,
                "The Pass Line contract follows the new table point.",
            )
        if roll.total == snapshot.point:
            return cls._contract_win(
                bet,
                bet.amount,
                "Pass Line wins when the table point repeats.",
            )
        if roll.total == 7:
            return cls._loss(bet, "Pass Line loses on the seven-out.")
        return cls._no_action(bet)

    @classmethod
    def _dont_pass(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve Don’t Pass behavior.

        Honor the table's configured bar number on the come-out roll.
        """
        if snapshot.phase == Phase.COME_OUT:
            if roll.total == rules.dont_bar:
                return cls._push(
                    bet,
                    "The configured Don't bar number is a standoff.",
                )
            if roll.total in CRAPS_NUMBERS:
                return cls._contract_win(
                    bet,
                    bet.amount,
                    "Don't Pass wins on a non-bar craps total.",
                )
            if roll.total in NATURALS:
                return cls._loss(
                    bet,
                    "Don't Pass loses on 7 or 11.",
                )
            return cls._no_action(
                bet,
                "The Don't Pass contract follows the table point.",
            )
        if roll.total == 7:
            return cls._contract_win(
                bet,
                bet.amount,
                "Don't Pass wins on the seven-out.",
            )
        if roll.total == snapshot.point:
            return cls._loss(
                bet,
                "Don't Pass loses when the table point repeats.",
            )
        return cls._no_action(bet)

    @classmethod
    def _come(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Come wager in the Come box or on its traveled number."""
        del snapshot, rules
        if bet.number is None:
            if roll.total in NATURALS:
                return cls._contract_win(
                    bet,
                    bet.amount,
                    "Come wins on its personal natural.",
                )
            if roll.total in CRAPS_NUMBERS:
                return cls._loss(
                    bet,
                    "Come loses on its personal come-out craps.",
                )
            return cls._travel(
                bet,
                roll.total,
                "Come travels from the Come box to a number.",
            )
        if roll.total == bet.number:
            return cls._contract_win(
                bet,
                bet.amount,
                "Come wins when its number repeats.",
            )
        if roll.total == 7:
            return cls._loss(bet, "Come loses when seven precedes its number.")
        return cls._no_action(bet)

    @classmethod
    def _dont_come(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Don’t Come wager before or after it travels."""
        del snapshot
        if bet.number is None:
            if roll.total == rules.dont_bar:
                return cls._push(
                    bet,
                    "Don't Come pushes on the configured bar number.",
                )
            if roll.total in CRAPS_NUMBERS:
                return cls._contract_win(
                    bet,
                    bet.amount,
                    "Don't Come wins on a non-bar craps total.",
                )
            if roll.total in NATURALS:
                return cls._loss(bet, "Don't Come loses on 7 or 11.")
            return cls._travel(
                bet,
                roll.total,
                "Don't Come travels behind its number.",
            )
        if roll.total == 7:
            return cls._contract_win(
                bet,
                bet.amount,
                "Don't Come wins when seven precedes its number.",
            )
        if roll.total == bet.number:
            return cls._loss(
                bet,
                "Don't Come loses when its number repeats.",
            )
        return cls._no_action(bet)

    @classmethod
    def _take_odds(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve positive odds attached to Pass Line or Come contracts."""
        parent = snapshot.bets.get(bet.parent_id or -1)
        if parent is None or bet.number is None:
            return Settlement(
                bet.bet_id,
                Outcome.RETURNED_OFF,
                credit=bet.amount,
                keep_active=False,
                note="Orphaned odds are safely returned.",
            )
        working = True
        if (
            snapshot.phase == Phase.COME_OUT
            and parent.kind == BetKind.COME
        ):
            working = rules.come_odds_working_on_comeout
        parent_resolves = roll.total in (7, bet.number)
        if not working:
            if parent_resolves:
                return Settlement(
                    bet.bet_id,
                    Outcome.RETURNED_OFF,
                    credit=bet.amount,
                    keep_active=False,
                    note="Come odds were off and are returned.",
                )
            return cls._no_action(bet, "Come odds are off on the come-out.")
        if roll.total == bet.number:
            profit = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.TRUE_WIN_PROFIT[bet.number],
                rules,
            )
            return cls._contract_win(
                bet,
                profit,
                "Taking odds wins at true odds.",
            )
        if roll.total == 7:
            return cls._loss(bet, "Taking odds loses to seven.")
        return cls._no_action(bet)

    @classmethod
    def _lay_odds(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve lay odds on Don’t Pass or Don’t Come contracts.

        Respect the configured working state for Don’t Come odds.
        """
        parent = snapshot.bets.get(bet.parent_id or -1)
        if parent is None or bet.number is None:
            return Settlement(
                bet.bet_id,
                Outcome.RETURNED_OFF,
                credit=bet.amount,
                keep_active=False,
                note="Orphaned lay odds are safely returned.",
            )
        working = True
        if (
            snapshot.phase == Phase.COME_OUT
            and parent.kind == BetKind.DONT_COME
        ):
            working = rules.dont_come_odds_working_on_comeout
        parent_resolves = roll.total in (7, bet.number)
        if not working:
            if parent_resolves:
                return Settlement(
                    bet.bet_id,
                    Outcome.RETURNED_OFF,
                    credit=bet.amount,
                    keep_active=False,
                    note="Don't Come odds were off and are returned.",
                )
            return cls._no_action(
                bet,
                "Don't Come odds are off on the come-out.",
            )
        if roll.total == 7:
            profit = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.TRUE_LAY_PROFIT[bet.number],
                rules,
            )
            return cls._contract_win(
                bet,
                profit,
                "Laying odds wins at true odds.",
            )
        if roll.total == bet.number:
            return cls._loss(bet, "Laying odds loses when its number rolls.")
        return cls._no_action(bet)

    @classmethod
    def _place(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Place wager while respecting its working state."""
        if not cls._working(
            bet,
            snapshot,
            rules.place_working_on_comeout,
        ):
            return cls._no_action(bet, "Place bet is off on the come-out.")
        if roll.total == bet.number:
            profit = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.PLACE_PROFIT[bet.number or 0],
                rules,
            )
            return cls._persistent_win(
                bet,
                profit,
                "Place bet wins and remains on the layout.",
            )
        if roll.total == 7:
            return cls._loss(bet, "Place bet loses to seven.")
        return cls._no_action(bet)

    @classmethod
    def _buy(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Buy wager at true odds with the configured commission."""
        if not cls._working(
            bet,
            snapshot,
            rules.buy_working_on_comeout,
        ):
            return cls._no_action(bet, "Buy bet is off on the come-out.")
        if roll.total == bet.number:
            gross = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.TRUE_WIN_PROFIT[bet.number or 0],
                rules,
            )
            commission = Decimal("0")
            if rules.buy_vig_timing == "on_win":
                commission = PayoutMath.vig(
                    bet.amount,
                    gross,
                    rules.buy_vig_rate,
                    rules.buy_vig_basis,
                    rules,
                )
            return cls._persistent_win(
                bet,
                gross,
                "Buy bet wins at true odds, less configured vig.",
                commission,
            )
        if roll.total == 7:
            return cls._loss(bet, "Buy bet loses to seven.")
        return cls._no_action(bet)

    @classmethod
    def _lay(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Lay wager at true odds with the configured commission."""
        if not cls._working(
            bet,
            snapshot,
            rules.lay_working_on_comeout,
        ):
            return cls._no_action(bet, "Lay bet is off on the come-out.")
        if roll.total == 7:
            gross = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.TRUE_LAY_PROFIT[bet.number or 0],
                rules,
            )
            commission = Decimal("0")
            if rules.lay_vig_timing == "on_win":
                commission = PayoutMath.vig(
                    bet.amount,
                    gross,
                    rules.lay_vig_rate,
                    rules.lay_vig_basis,
                    rules,
                )
            return cls._persistent_win(
                bet,
                gross,
                "Lay bet wins on seven, less configured vig.",
                commission,
            )
        if roll.total == bet.number:
            return cls._loss(bet, "Lay bet loses when its number appears.")
        return cls._no_action(bet)

    @classmethod
    def _field(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve the one-roll Field wager.

        Use the table's configured payouts for totals of 2 and 12.
        """
        del snapshot
        if roll.total not in (2, 3, 4, 9, 10, 11, 12):
            return cls._loss(bet, "Field loses on 5, 6, 7, or 8.")
        ratio = Decimal("1")
        if roll.total == 2:
            ratio = rules.field_2_profit
        elif roll.total == 12:
            ratio = rules.field_12_profit
        profit = PayoutMath.gross_profit(bet.amount, ratio, rules)
        return cls._contract_win(bet, profit, "Field resolves in one roll.")

    @classmethod
    def _big_6(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Big 6 wager through the shared even-money rule."""
        return cls._big_number(bet, snapshot, roll, rules, 6)

    @classmethod
    def _big_8(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Big 8 wager through the shared even-money rule."""
        return cls._big_number(bet, snapshot, roll, rules, 8)

    @classmethod
    def _big_number(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
        number: int,
    ) -> Settlement:
        """Resolve persistent Big 6 or Big 8 behavior for a selected number."""
        if not cls._working(
            bet,
            snapshot,
            rules.big_6_8_working_on_comeout,
        ):
            return cls._no_action(bet, "Big 6/8 is off on the come-out.")
        if roll.total == number:
            return cls._persistent_win(
                bet,
                bet.amount,
                f"Big {number} wins even money and remains up.",
            )
        if roll.total == 7:
            return cls._loss(bet, f"Big {number} loses to seven.")
        return cls._no_action(bet)

    @classmethod
    def _hardway(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a Hardway against the hard total, easy total, or seven."""
        if not cls._working(
            bet,
            snapshot,
            rules.hardways_working_on_comeout,
        ):
            return cls._no_action(bet, "Hardway is off on the come-out.")
        if roll.total == bet.number and roll.is_pair:
            profit = PayoutMath.gross_profit(
                bet.amount,
                PayoutMath.HARDWAY_PROFIT[bet.number or 0],
                rules,
            )
            return cls._persistent_win(
                bet,
                profit,
                "Hardway wins as a pair and remains up.",
            )
        if roll.total == 7:
            return cls._loss(bet, "Hardway loses to seven.")
        if roll.total == bet.number and not roll.is_pair:
            return cls._loss(bet, "Hardway loses to the easy combination.")
        return cls._no_action(bet)

    @classmethod
    def _proposition(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve a supported single-total or grouped one-roll proposition."""
        del snapshot
        winners = {
            BetKind.ANY_SEVEN: roll.total == 7,
            BetKind.ANY_CRAPS: roll.total in CRAPS_NUMBERS,
            BetKind.ACES: roll.total == 2,
            BetKind.ACE_DEUCE: roll.total == 3,
            BetKind.YO: roll.total == 11,
            BetKind.BOXCARS: roll.total == 12,
        }
        if not winners[bet.kind]:
            return cls._loss(bet, "One-roll proposition misses.")
        profit = PayoutMath.gross_profit(
            bet.amount,
            PayoutMath.PROPOSITION_PROFIT[bet.kind],
            rules,
        )
        return cls._contract_win(
            bet,
            profit,
            "One-roll proposition wins.",
        )

    @classmethod
    def _horn(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve the winning quarter of a Horn wager.

        The other three component stakes lose on the same roll.
        """
        del snapshot
        if roll.total not in (2, 3, 11, 12):
            return cls._loss(bet, "All four Horn components lose.")
        component = bet.amount / Decimal("4")
        ratio = Decimal("30") if roll.total in (2, 12) else Decimal("15")
        profit = PayoutMath.gross_profit(component, ratio, rules)
        credit = component + profit
        return Settlement(
            bet.bet_id,
            Outcome.WIN,
            credit=credit,
            profit=credit - bet.amount,
            keep_active=False,
            note="Horn pays one quarter; the other three quarters lose.",
        )

    @classmethod
    def _c_and_e(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve the Craps or Eleven half of a split C & E wager."""
        del snapshot
        half = bet.amount / Decimal("2")
        if roll.total in CRAPS_NUMBERS:
            profit = PayoutMath.gross_profit(half, Decimal("7"), rules)
            credit = half + profit
        elif roll.total == 11:
            profit = PayoutMath.gross_profit(half, Decimal("15"), rules)
            credit = half + profit
        else:
            return cls._loss(bet, "Both halves of C & E lose.")
        return Settlement(
            bet.bet_id,
            Outcome.WIN,
            credit=credit,
            profit=credit - bet.amount,
            keep_active=False,
            note="C & E pays its winning half; the other half loses.",
        )

    @classmethod
    def _world(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve the winning fifth of a World wager.

        The other four component stakes lose on the same roll.
        """
        del snapshot
        fifth = bet.amount / Decimal("5")
        if roll.total == 7:
            profit = PayoutMath.gross_profit(fifth, Decimal("4"), rules)
            credit = fifth + profit
        elif roll.total in (2, 3, 11, 12):
            ratio = (
                Decimal("30")
                if roll.total in (2, 12)
                else Decimal("15")
            )
            profit = PayoutMath.gross_profit(fifth, ratio, rules)
            credit = fifth + profit
        else:
            return cls._loss(bet, "All five World components lose.")
        return Settlement(
            bet.bet_id,
            Outcome.WIN,
            credit=credit,
            profit=credit - bet.amount,
            keep_active=False,
            note="World pays one fifth; the other four fifths lose.",
        )

    @classmethod
    def _hop(
        cls,
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        roll: DiceRoll,
        rules: TableRules,
    ) -> Settlement:
        """Resolve an exact unordered two-die Hop combination."""
        del snapshot
        if bet.hop_pair != roll.unordered_pair:
            return cls._loss(bet, "The selected Hop combination misses.")
        ratio = Decimal("30") if roll.is_pair else Decimal("15")
        profit = PayoutMath.gross_profit(bet.amount, ratio, rules)
        return cls._contract_win(
            bet,
            profit,
            "Hop combination wins in one roll.",
        )

    @staticmethod
    def _working(
        bet: BetSnapshot,
        snapshot: TableSnapshot,
        default_on_comeout: bool,
    ) -> bool:
        """Determine whether a multi-roll wager is active for this phase."""
        if bet.working is not None:
            return bet.working
        if snapshot.phase == Phase.POINT_ON:
            return True
        return default_on_comeout


# ---------------------------------------------------------------------------
# VIEW
# ---------------------------------------------------------------------------


class CrapsView(ABC):
    """Output boundary; simulation logic never prints directly."""

    @abstractmethod
    def on_message(self, message: str) -> None:
        """Display one human-facing message."""

    @abstractmethod
    def on_roll(self, event: RollEvent, model: TableModel) -> None:
        """Display one completed roll."""

    @abstractmethod
    def on_session_end(self, result: SessionResult) -> None:
        """Display a final session summary."""


class NullView(CrapsView):
    """A silent view for tests and high-volume Monte Carlo runs."""

    def on_message(self, message: str) -> None:
        """Discard a strategy message during silent execution."""
        del message

    def on_roll(self, event: RollEvent, model: TableModel) -> None:
        """Discard a completed roll event during silent execution."""
        del event, model

    def on_session_end(self, result: SessionResult) -> None:
        """Discard the final result during silent execution."""
        del result


class ConsoleView(CrapsView):
    """Readable audit output without contaminating the model or controller."""

    def on_message(self, message: str) -> None:
        """Print a strategy or controller message to standard output."""
        print(message)

    def on_roll(self, event: RollEvent, model: TableModel) -> None:
        """Print one roll, point transition, balances, and settlements."""
        point_before = event.point_before or "OFF"
        point_after = event.point_after or "OFF"
        print(
            f"Roll {model.roll_number:>4}: "
            f"{event.roll.die_1}-{event.roll.die_2} "
            f"= {event.roll.total:>2} | "
            f"point {point_before} -> {point_after} | "
            f"cash {money_text(model.ledger.cash)} | "
            f"equity {money_text(model.equity)}"
        )
        for settlement in event.settlements:
            if settlement.outcome != Outcome.NO_ACTION:
                print(
                    f"    bet #{settlement.bet_id}: "
                    f"{settlement.outcome.value}; "
                    f"credit {money_text(settlement.credit)}; "
                    f"{settlement.note}"
                )

    def on_session_end(self, result: SessionResult) -> None:
        """Print the final session summary."""
        print("\nSession complete")
        print(f"  Rolls:          {result.rolls}")
        print(f"  Hands:          {result.hands}")
        print(f"  Closing cash:   {money_text(result.closing_cash)}")
        print(f"  Closing equity: {money_text(result.closing_equity)}")
        print(f"  Net:            {money_text(result.net)}")
        print(f"  Stop reason:    {result.stop_reason}")


# ---------------------------------------------------------------------------
# CONTROLLER: TABLE OPERATIONS AND STATE TRANSITIONS
# ---------------------------------------------------------------------------


class CrapsController:
    """The sole authority permitted to mutate TableModel."""

    def __init__(
        self,
        model: TableModel,
        dice: DiceSource,
        view: Optional[CrapsView] = None,
    ) -> None:
        """Bind the mutable table model, dice source, and output view."""
        self.model = model
        self.dice = dice
        self.view = view or NullView()

    def place_bet(
        self,
        kind: BetKind,
        amount: Any,
        number: Optional[int] = None,
        parent_id: Optional[int] = None,
        hop_pair: Optional[Sequence[int]] = None,
        working: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Bet:
        """Validate, reserve, and register a new wager atomically."""
        wager = as_money(amount)
        normalized_hop = None
        if hop_pair is not None:
            if len(hop_pair) != 2:
                raise BetError("A Hop wager needs exactly two die faces.")
            normalized_hop = tuple(
                sorted((int(hop_pair[0]), int(hop_pair[1])))
            )
        self._validate_new_bet(
            kind,
            wager,
            number,
            parent_id,
            normalized_hop,
        )
        upfront_vig = self._upfront_vig(kind, wager, number)
        total_debit = wager + upfront_vig
        self.model.ledger.debit(
            total_debit,
            self.model.roll_number,
            f"Place {kind.value} stake {money_text(wager)}"
            + (
                f" plus vig {money_text(upfront_vig)}"
                if upfront_vig
                else ""
            ),
        )
        bet = Bet(
            bet_id=self.model.next_bet_id,
            kind=kind,
            amount=wager,
            number=number,
            parent_id=parent_id,
            hop_pair=normalized_hop,
            working=working,
            created_roll=self.model.roll_number,
            metadata=metadata or {},
        )
        self.model.bets[bet.bet_id] = bet
        self.model.next_bet_id += 1
        return bet

    def ensure_bet(self, **kwargs: Any) -> Bet:
        """Return an existing exact match or place one new wager."""
        kind = kwargs["kind"]
        number = kwargs.get("number")
        parent_id = kwargs.get("parent_id")
        hop_pair = kwargs.get("hop_pair")
        normalized_hop = None
        if hop_pair is not None:
            normalized_hop = tuple(sorted(map(int, hop_pair)))
        for bet in self.model.active_bets(kind, number, parent_id):
            if bet.hop_pair == normalized_hop:
                return bet
        return self.place_bet(**kwargs)

    def take_odds(
        self,
        parent_id: int,
        amount: Any = "max",
    ) -> Bet:
        """Attach one odds wager to an eligible contract wager."""
        parent = self._active_bet(parent_id)
        if parent.kind not in {
            BetKind.PASS_LINE,
            BetKind.DONT_PASS,
            BetKind.COME,
            BetKind.DONT_COME,
        }:
            raise BetError("Odds require a line, Come, or Don't Come parent.")
        number = self._parent_number(parent)
        if number is None:
            raise BetError("Odds cannot be placed until a point exists.")
        odds_kind = (
            BetKind.TAKE_ODDS
            if parent.kind in {BetKind.PASS_LINE, BetKind.COME}
            else BetKind.LAY_ODDS
        )
        existing = self.model.active_bets(odds_kind, parent_id=parent_id)
        if existing:
            return existing[0]
        schedule = (
            self.model.rules.odds.take
            if odds_kind == BetKind.TAKE_ODDS
            else self.model.rules.odds.lay
        )
        maximum = self.model.rules.round_money(
            parent.amount * schedule[number]
        )
        wager = maximum if amount == "max" else as_money(amount)
        if wager > maximum:
            raise BetError(
                f"Odds of {money_text(wager)} exceed the configured maximum "
                f"of {money_text(maximum)}."
            )
        return self.place_bet(
            odds_kind,
            wager,
            number=number,
            parent_id=parent_id,
        )

    def remove_bet(self, bet_id: int) -> Bet:
        """Return an active removable stake to cash."""
        bet = self._active_bet(bet_id)
        if bet.kind in {BetKind.PASS_LINE, BetKind.COME}:
            if self._parent_number(bet) is not None:
                raise BetError("A positive contract wager cannot be removed.")
        if bet.kind in {BetKind.TAKE_ODDS, BetKind.LAY_ODDS}:
            pass
        self.model.ledger.credit(
            bet.amount,
            self.model.roll_number,
            f"Remove bet #{bet.bet_id} ({bet.kind.value})",
        )
        bet.status = BetStatus.REMOVED
        self._return_child_odds(bet.bet_id)
        return bet

    def reduce_bet(self, bet_id: int, new_amount: Any) -> Bet:
        """Reduce a removable wager and return the difference to cash."""
        bet = self._active_bet(bet_id)
        target = as_money(new_amount)
        if target <= 0 or target >= bet.amount:
            raise BetError("A reduction must be positive and below the stake.")
        self._validate_amount(bet.kind, target, bet.number)
        if bet.kind in {BetKind.PASS_LINE, BetKind.COME}:
            if self._parent_number(bet) is not None:
                raise BetError("A positive contract wager cannot be reduced.")
        difference = bet.amount - target
        bet.amount = target
        self.model.ledger.credit(
            difference,
            self.model.roll_number,
            f"Reduce bet #{bet.bet_id}",
        )
        return bet

    def increase_bet(self, bet_id: int, new_amount: Any) -> Bet:
        """Increase a legal wager by reserving only the added stake."""
        bet = self._active_bet(bet_id)
        target = as_money(new_amount)
        if target <= bet.amount:
            raise BetError("An increase must exceed the existing stake.")
        if bet.kind in {BetKind.DONT_PASS, BetKind.DONT_COME}:
            if self._parent_number(bet) is not None:
                raise BetError(
                    "A Don't contract cannot increase after travel."
                )
        if bet.kind in {BetKind.PASS_LINE, BetKind.COME}:
            if self._parent_number(bet) is not None:
                raise BetError(
                    "A positive contract cannot increase after travel."
                )
        self._validate_amount(bet.kind, target, bet.number)
        difference = target - bet.amount
        old_vig = self._upfront_vig(
            bet.kind,
            bet.amount,
            bet.number,
        )
        new_vig = self._upfront_vig(
            bet.kind,
            target,
            bet.number,
        )
        added_vig = max(new_vig - old_vig, Decimal("0"))
        self.model.ledger.debit(
            difference + added_vig,
            self.model.roll_number,
            f"Increase bet #{bet.bet_id}"
            + (
                f" plus vig {money_text(added_vig)}"
                if added_vig
                else ""
            ),
        )
        bet.amount = target
        return bet

    def set_working(self, bet_id: int, working: Optional[bool]) -> Bet:
        """Override or restore the table default for a multi-roll wager."""
        bet = self._active_bet(bet_id)
        if bet.kind not in {
            BetKind.PLACE,
            BetKind.BUY,
            BetKind.LAY,
            BetKind.BIG_6,
            BetKind.BIG_8,
            BetKind.HARDWAY,
        }:
            raise BetError("Working state applies only to multi-roll wagers.")
        bet.working = working
        return bet

    def regress_bets(
        self,
        kind: BetKind,
        factor: Any,
        minimums: Optional[Mapping[int, Any]] = None,
    ) -> list[Bet]:
        """Scale matching wagers down and return every removed chip."""
        ratio = as_money(factor)
        if ratio <= 0 or ratio >= 1:
            raise BetError("Regression factor must be greater than 0 and < 1.")
        changed: list[Bet] = []
        for bet in self.model.active_bets(kind):
            floor = self.model.rules.minimum_for(kind, bet.number)
            if minimums and bet.number in minimums:
                floor = as_money(minimums[bet.number])
            target = self.model.rules.round_money(bet.amount * ratio)
            target = max(target, floor)
            increment = self.model.rules.increment_for(kind, bet.number)
            target = self._floor_to_increment(target, increment)
            target = max(target, floor)
            if target < bet.amount:
                self.reduce_bet(bet.bet_id, target)
                changed.append(bet)
        return changed

    def press_bet(
        self,
        bet_id: int,
        amount: Optional[Any] = None,
        factor: Optional[Any] = None,
        profit_fraction: Optional[Any] = None,
        last_profit: Optional[Money] = None,
    ) -> Bet:
        """Increase a persistent wager using one explicit press policy."""
        bet = self._active_bet(bet_id)
        modes = sum(
            value is not None
            for value in (amount, factor, profit_fraction)
        )
        if modes != 1:
            raise BetError("Press requires exactly one sizing mode.")
        if amount is not None:
            target = bet.amount + as_money(amount)
        elif factor is not None:
            target = self.model.rules.round_money(
                bet.amount * as_money(factor)
            )
        else:
            if last_profit is None:
                raise BetError("Profit-fraction press needs a winning profit.")
            target = bet.amount + self.model.rules.round_money(
                last_profit * as_money(profit_fraction)
            )
        increment = self.model.rules.increment_for(bet.kind, bet.number)
        target = self._floor_to_increment(target, increment)
        if target <= bet.amount:
            return bet
        return self.increase_bet(bet.bet_id, target)

    def roll_once(self) -> RollEvent:
        """Roll, settle from a snapshot, update the puck, and publish."""
        if self.model.session_stopped:
            raise CrapsError("The session has already been stopped.")
        self.model.roll_number += 1
        self.model.hand_roll_number += 1
        snapshot = self.model.snapshot()
        roll = self.dice.roll()
        settlements = BetResolver.resolve_all(
            snapshot,
            roll,
            self.model.rules,
        )
        for settlement in settlements:
            self._apply_settlement(settlement)
        point_established = False
        point_made = False
        seven_out = False
        if snapshot.phase == Phase.COME_OUT:
            if roll.total in POINT_NUMBERS:
                self.model.point = roll.total
                point_established = True
        else:
            if roll.total == snapshot.point:
                self.model.point = None
                point_made = True
            elif roll.total == 7:
                self.model.point = None
                seven_out = True
        self._clean_orphaned_odds()
        event = RollEvent(
            roll=roll,
            phase_before=snapshot.phase,
            point_before=snapshot.point,
            phase_after=self.model.phase,
            point_after=self.model.point,
            settlements=tuple(settlements),
            point_established=point_established,
            point_made=point_made,
            seven_out=seven_out,
        )
        self.view.on_roll(event, self.model)
        return event

    def start_new_hand(self) -> None:
        """Advance the shooter counter after a completed seven-out."""
        if self.model.point is not None:
            raise CrapsError("A new hand cannot begin while a point is on.")
        self.model.hand_number += 1
        self.model.hand_roll_number = 0

    def stop(self, reason: str) -> None:
        """Mark the session stopped and record the reason."""
        self.model.session_stopped = True
        self.model.stop_reason = reason

    def _apply_settlement(self, settlement: Settlement) -> None:
        """Apply one settlement to cash, wager state, and traveled numbers."""
        bet = self.model.bets[settlement.bet_id]
        if settlement.credit:
            self.model.ledger.credit(
                settlement.credit,
                self.model.roll_number,
                f"Settle bet #{bet.bet_id}: {settlement.outcome.value}",
            )
        if settlement.travel_to is not None:
            bet.number = settlement.travel_to
        if not settlement.keep_active:
            bet.status = BetStatus.RESOLVED
        if settlement.outcome in {
            Outcome.LOSS,
            Outcome.WIN,
            Outcome.RETURNED_OFF,
        } and not settlement.keep_active:
            self._resolve_child_odds_if_orphaned(bet.bet_id)

    def _validate_new_bet(
        self,
        kind: BetKind,
        amount: Money,
        number: Optional[int],
        parent_id: Optional[int],
        hop_pair: Optional[tuple[int, int]],
    ) -> None:
        """Validate wager type, amount, number, parent, and table phase."""
        self._validate_amount(kind, amount, number)
        numbered = {
            BetKind.PLACE,
            BetKind.BUY,
            BetKind.LAY,
            BetKind.HARDWAY,
        }
        if kind in numbered and number not in POINT_NUMBERS:
            raise BetError(f"{kind.value} requires a box number.")
        if kind == BetKind.HARDWAY and number not in (4, 6, 8, 10):
            raise BetError("Hardways exist only on 4, 6, 8, and 10.")
        if kind in {BetKind.BIG_6, BetKind.BIG_8} and number is not None:
            raise BetError("Big 6/8 does not accept a separate number.")
        if kind == BetKind.HOP:
            if hop_pair is None:
                raise BetError("Hop requires a two-die combination.")
            if any(face not in range(1, 7) for face in hop_pair):
                raise BetError("Hop die faces must be from 1 through 6.")
        elif hop_pair is not None:
            raise BetError("Only a Hop wager accepts hop_pair.")
        if kind in {BetKind.TAKE_ODDS, BetKind.LAY_ODDS}:
            if parent_id is None:
                raise BetError("Odds require parent_id.")
            parent = self._active_bet(parent_id)
            positive_parent = parent.kind in {
                BetKind.PASS_LINE,
                BetKind.COME,
            }
            expected_kind = (
                BetKind.TAKE_ODDS
                if positive_parent
                else BetKind.LAY_ODDS
            )
            if parent.kind not in {
                BetKind.PASS_LINE,
                BetKind.DONT_PASS,
                BetKind.COME,
                BetKind.DONT_COME,
            }:
                raise BetError("Odds parent is not a contract wager.")
            if kind != expected_kind:
                raise BetError("Odds direction does not match its parent.")
            expected_number = self._parent_number(parent)
            if expected_number is None:
                raise BetError("Odds require an established contract point.")
            if number != expected_number:
                raise BetError("Odds number must match the parent contract.")
            if self.model.active_bets(kind, parent_id=parent_id):
                raise BetError("The parent already has an odds wager.")
            schedule = (
                self.model.rules.odds.take
                if kind == BetKind.TAKE_ODDS
                else self.model.rules.odds.lay
            )
            maximum = self.model.rules.round_money(
                parent.amount * schedule[expected_number]
            )
            if amount > maximum:
                raise BetError(
                    f"Odds exceed the configured maximum of "
                    f"{money_text(maximum)}."
                )
        elif parent_id is not None:
            raise BetError("parent_id is reserved for odds wagers.")
        if kind in {BetKind.PASS_LINE, BetKind.DONT_PASS}:
            if self.model.phase != Phase.COME_OUT:
                raise BetError("Line wagers must be made on a come-out roll.")
        if kind in {BetKind.COME, BetKind.DONT_COME}:
            if self.model.phase != Phase.POINT_ON:
                raise BetError("Come wagers require an established point.")

    def _validate_amount(
        self,
        kind: BetKind,
        amount: Money,
        number: Optional[int],
    ) -> None:
        """Enforce positive stakes, table minimums, and legal increments."""
        if amount <= 0:
            raise BetError("Wager amount must be positive.")
        minimum = self.model.rules.minimum_for(kind, number)
        if amount < minimum:
            raise BetError(
                f"{kind.value} minimum is {money_text(minimum)}."
            )
        if self.model.rules.enforce_increments:
            increment = self.model.rules.increment_for(kind, number)
            if amount % increment != 0:
                raise BetError(
                    f"{kind.value} must be in {money_text(increment)} "
                    "increments."
                )

    def _upfront_vig(
        self,
        kind: BetKind,
        amount: Money,
        number: Optional[int],
    ) -> Money:
        """Calculate commission due when a Buy or Lay wager is placed."""
        rules = self.model.rules
        if kind == BetKind.BUY and rules.buy_vig_timing == "upfront":
            gross = PayoutMath.gross_profit(
                amount,
                PayoutMath.TRUE_WIN_PROFIT[number or 0],
                rules,
            )
            return PayoutMath.vig(
                amount,
                gross,
                rules.buy_vig_rate,
                rules.buy_vig_basis,
                rules,
            )
        if kind == BetKind.LAY and rules.lay_vig_timing == "upfront":
            gross = PayoutMath.gross_profit(
                amount,
                PayoutMath.TRUE_LAY_PROFIT[number or 0],
                rules,
            )
            return PayoutMath.vig(
                amount,
                gross,
                rules.lay_vig_rate,
                rules.lay_vig_basis,
                rules,
            )
        return Decimal("0")

    def _parent_number(self, parent: Bet) -> Optional[int]:
        """Return the effective point associated with a contract wager."""
        if parent.kind in {BetKind.PASS_LINE, BetKind.DONT_PASS}:
            return self.model.point
        return parent.number

    def _active_bet(self, bet_id: int) -> Bet:
        """Return an active wager by identifier or raise a wager error."""
        bet = self.model.bets.get(bet_id)
        if bet is None or not bet.is_active:
            raise BetError(f"Bet #{bet_id} is not active.")
        return bet

    def _return_child_odds(self, parent_id: int) -> None:
        """Return active odds attached to a removed parent contract."""
        for kind in (BetKind.TAKE_ODDS, BetKind.LAY_ODDS):
            for child in self.model.active_bets(kind, parent_id=parent_id):
                self.model.ledger.credit(
                    child.amount,
                    self.model.roll_number,
                    f"Return odds bet #{child.bet_id}",
                )
                child.status = BetStatus.REMOVED

    def _resolve_child_odds_if_orphaned(self, parent_id: int) -> None:
        """Defer orphan cleanup until all snapshot settlements are applied."""
        del parent_id
        # Child odds have their own settlement in the same immutable snapshot.
        # Delaying orphan cleanup until every settlement is applied preserves
        # true-odds wins on the exact roll that resolves the parent contract.

    def _clean_orphaned_odds(self) -> None:
        """Return any active odds whose parent contract has resolved."""
        for kind in (BetKind.TAKE_ODDS, BetKind.LAY_ODDS):
            for odds in list(self.model.active_bets(kind)):
                parent = self.model.bets.get(odds.parent_id or -1)
                if parent is None or not parent.is_active:
                    self.model.ledger.credit(
                        odds.amount,
                        self.model.roll_number,
                        f"Return orphaned odds bet #{odds.bet_id}",
                    )
                    odds.status = BetStatus.RESOLVED

    @staticmethod
    def _floor_to_increment(value: Money, increment: Money) -> Money:
        """Round a wager amount down to a legal chip increment."""
        return (value / increment).quantize(
            Decimal("1"),
            rounding=ROUND_DOWN,
        ) * increment


# ---------------------------------------------------------------------------
# YAML CONFIGURATION: STRICT PARSING, NO SILENTLY IGNORED KEYS
# ---------------------------------------------------------------------------


class StrictMapping:
    """Utility that consumes known keys and rejects every leftover key."""

    def __init__(self, data: Mapping[str, Any], path: str) -> None:
        """Copy a mapping and remember its configuration path."""
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"{path} must be a mapping.")
        self._data = dict(data)
        self.path = path

    def pop(self, key: str, default: Any = None) -> Any:
        """Consume an optional key and return its value or a default."""
        return self._data.pop(key, default)

    def require(self, key: str) -> Any:
        """Consume a required key or raise a configuration error."""
        if key not in self._data:
            raise ConfigurationError(
                f"Missing required key {self.path}.{key}."
            )
        return self._data.pop(key)

    def finish(self) -> None:
        """Reject any keys that were not explicitly consumed."""
        if self._data:
            unknown = ", ".join(sorted(self._data))
            raise ConfigurationError(
                f"Unknown key(s) at {self.path}: {unknown}."
            )


@dataclass(frozen=True)
class SessionConfig:
    """Session bankroll, stopping limits, roll limits, and random seed."""
    bankroll: Money
    target: Money
    stop_loss: Money
    max_rolls: int
    max_hands: int
    seed: Optional[int]


@dataclass(frozen=True)
class ActionSpec:
    """One validated strategy action and the event that triggers it."""
    action_id: str
    trigger: str
    conditions: Mapping[str, Any]
    operation: Mapping[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    """A deterministic YAML scenario with setup, rolls, and assertions."""
    name: str
    bankroll: Money
    setup: tuple[Mapping[str, Any], ...]
    rolls: tuple[tuple[int, int], ...]
    expected: Mapping[str, Any]


@dataclass(frozen=True)
class EngineConfig:
    """Complete immutable configuration produced by the YAML loader."""
    version: int
    name: str
    description: str
    session: SessionConfig
    rules: TableRules
    actions: tuple[ActionSpec, ...]
    scenarios: tuple[ScenarioSpec, ...]


class YamlConfigLoader:
    """Load the documented schema and reject configuration drift."""

    ROOT_KEYS = {
        "version",
        "meta",
        "session",
        "rules",
        "strategy",
        "scenarios",
    }

    ACTION_TRIGGERS = {
        "session_start",
        "hand_start",
        "before_roll",
        "come_out",
        "after_roll",
        "point_established",
        "point_made",
        "seven_out",
        "bet_win",
        "bet_loss",
    }

    ACTION_TYPES = {
        "place_bet",
        "ensure_bet",
        "take_odds",
        "remove_bets",
        "regress_bets",
        "press_bet",
        "set_working",
        "stop_session",
        "message",
    }

    CONDITION_KEYS = {
        "phase",
        "point_in",
        "roll_total_in",
        "roll_number_gte",
        "roll_number_lte",
        "roll_every",
        "hand_number_gte",
        "hand_number_lte",
        "cash_gte",
        "cash_lte",
        "equity_gte",
        "equity_lte",
        "bet_exists",
        "bet_not_exists",
        "winning_kind",
        "losing_kind",
    }

    @classmethod
    def load(cls, path: Path | str) -> EngineConfig:
        """Read YAML from disk and convert it into validated engine config."""
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EngineConfig:
        """Parse and validate a configuration mapping in memory."""
        root = StrictMapping(raw, "root")
        version = int(root.require("version"))
        if version != 1:
            raise ConfigurationError(
                "Only YAML schema version 1 is supported."
            )
        meta = cls._parse_meta(root.require("meta"))
        session = cls._parse_session(root.require("session"))
        rules = cls._parse_rules(root.require("rules"))
        actions = cls._parse_strategy(root.require("strategy"))
        scenarios = cls._parse_scenarios(root.pop("scenarios", []))
        root.finish()
        return EngineConfig(
            version=version,
            name=meta[0],
            description=meta[1],
            session=session,
            rules=rules,
            actions=actions,
            scenarios=scenarios,
        )

    @staticmethod
    def _parse_meta(raw: Mapping[str, Any]) -> tuple[str, str]:
        """Parse the project name and description metadata."""
        data = StrictMapping(raw, "meta")
        name = str(data.require("name"))
        description = str(data.require("description"))
        data.finish()
        return name, description

    @staticmethod
    def _parse_session(raw: Mapping[str, Any]) -> SessionConfig:
        """Parse bankroll limits, run limits, and the optional random seed."""
        data = StrictMapping(raw, "session")
        seed_raw = data.pop("seed", None)
        config = SessionConfig(
            bankroll=as_money(data.require("bankroll")),
            target=as_money(data.require("target")),
            stop_loss=as_money(data.require("stop_loss")),
            max_rolls=int(data.require("max_rolls")),
            max_hands=int(data.require("max_hands")),
            seed=None if seed_raw is None else int(seed_raw),
        )
        data.finish()
        if not (
            Decimal("0") <= config.stop_loss < config.target
            and config.bankroll > Decimal("0")
        ):
            raise ConfigurationError("Session money limits are contradictory.")
        if config.max_rolls <= 0 or config.max_hands <= 0:
            raise ConfigurationError("Session limits must be positive.")
        return config

    @classmethod
    def _parse_rules(cls, raw: Mapping[str, Any]) -> TableRules:
        """Parse table rules, odds schedules, minimums, and increments."""
        data = StrictMapping(raw, "rules")
        odds_raw = StrictMapping(data.require("odds"), "rules.odds")
        take = cls._point_money_map(odds_raw.require("take"), "odds.take")
        lay = cls._point_money_map(odds_raw.require("lay"), "odds.lay")
        odds_raw.finish()
        working = StrictMapping(
            data.require("working_on_comeout"),
            "rules.working_on_comeout",
        )
        place_working = bool(working.require("place"))
        buy_working = bool(working.require("buy"))
        lay_working = bool(working.require("lay"))
        hard_working = bool(working.require("hardways"))
        big_working = bool(working.require("big_6_8"))
        come_odds_working = bool(working.require("come_odds"))
        dont_odds_working = bool(working.require("dont_come_odds"))
        working.finish()
        buy_vig = StrictMapping(data.require("buy_vig"), "rules.buy_vig")
        buy_rate = as_money(buy_vig.require("rate"))
        buy_timing = str(buy_vig.require("timing"))
        buy_basis = str(buy_vig.require("basis"))
        buy_vig.finish()
        lay_vig = StrictMapping(data.require("lay_vig"), "rules.lay_vig")
        lay_rate = as_money(lay_vig.require("rate"))
        lay_timing = str(lay_vig.require("timing"))
        lay_basis = str(lay_vig.require("basis"))
        lay_vig.finish()
        minimums = cls._validate_bet_rule_map(
            data.require("minimums"),
            "rules.minimums",
        )
        increments = cls._validate_bet_rule_map(
            data.require("increments"),
            "rules.increments",
        )
        result = TableRules(
            dont_bar=int(data.require("dont_bar")),
            field_2_profit=as_money(data.require("field_2_profit")),
            field_12_profit=as_money(data.require("field_12_profit")),
            buy_vig_rate=buy_rate,
            buy_vig_timing=buy_timing,
            buy_vig_basis=buy_basis,
            lay_vig_rate=lay_rate,
            lay_vig_timing=lay_timing,
            lay_vig_basis=lay_basis,
            chip_unit=as_money(data.require("chip_unit")),
            payout_rounding=str(data.require("payout_rounding")),
            enforce_increments=bool(data.require("enforce_increments")),
            place_working_on_comeout=place_working,
            buy_working_on_comeout=buy_working,
            lay_working_on_comeout=lay_working,
            hardways_working_on_comeout=hard_working,
            big_6_8_working_on_comeout=big_working,
            come_odds_working_on_comeout=come_odds_working,
            dont_come_odds_working_on_comeout=dont_odds_working,
            odds=OddsSchedule(take=take, lay=lay),
            minimums=minimums,
            increments=increments,
        )
        data.finish()
        return result

    @staticmethod
    def _point_money_map(
        raw: Mapping[Any, Any],
        path: str,
    ) -> dict[int, Money]:
        """Parse a complete six-number map of Decimal values."""
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{path} must be a mapping.")
        result = {int(key): as_money(value) for key, value in raw.items()}
        if set(result) != set(POINT_NUMBERS):
            raise ConfigurationError(
                f"{path} must define all six point numbers exactly."
            )
        return result


    @staticmethod
    def _validate_bet_rule_map(
        raw: Mapping[str, Any],
        path: str,
    ) -> dict[str, Any]:
        """Validate wager-rule keys and normalize nested point maps."""
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{path} must be a mapping.")
        allowed = {kind.value for kind in BetKind}
        unknown = set(map(str, raw)) - allowed
        if unknown:
            raise ConfigurationError(
                f"Unknown wager key(s) at {path}: "
                + ", ".join(sorted(unknown))
            )
        result: dict[str, Any] = {}
        for key, value in raw.items():
            name = str(key)
            if isinstance(value, Mapping):
                allowed_numbers = {str(number) for number in POINT_NUMBERS}
                allowed_numbers.add("default")
                unknown_numbers = set(map(str, value)) - allowed_numbers
                if unknown_numbers:
                    raise ConfigurationError(
                        f"Unknown number key(s) at {path}.{name}: "
                        + ", ".join(sorted(unknown_numbers))
                    )
                converted = {
                    sub_key: as_money(sub_value)
                    for sub_key, sub_value in value.items()
                }
                if any(amount <= 0 for amount in converted.values()):
                    raise ConfigurationError(
                        f"All values at {path}.{name} must be positive."
                    )
                result[name] = converted
            else:
                amount = as_money(value)
                if amount <= 0:
                    raise ConfigurationError(
                        f"The value at {path}.{name} must be positive."
                    )
                result[name] = amount
        return result

    @classmethod
    def _parse_strategy(
        cls,
        raw: Mapping[str, Any],
    ) -> tuple[ActionSpec, ...]:
        """Parse trigger-driven strategy actions and reject unknown fields."""
        data = StrictMapping(raw, "strategy")
        raw_actions = data.require("actions")
        data.finish()
        if not isinstance(raw_actions, list):
            raise ConfigurationError("strategy.actions must be a list.")
        actions: list[ActionSpec] = []
        seen_ids: set[str] = set()
        for index, raw_action in enumerate(raw_actions):
            item = StrictMapping(raw_action, f"strategy.actions[{index}]")
            action_id = str(item.require("id"))
            if action_id in seen_ids:
                raise ConfigurationError(f"Duplicate action id: {action_id}.")
            seen_ids.add(action_id)
            trigger = str(item.require("trigger"))
            if trigger not in cls.ACTION_TRIGGERS:
                raise ConfigurationError(f"Unknown trigger: {trigger}.")
            conditions = item.pop("when", {})
            if not isinstance(conditions, Mapping):
                raise ConfigurationError("Action 'when' must be a mapping.")
            unknown_conditions = set(conditions) - cls.CONDITION_KEYS
            if unknown_conditions:
                raise ConfigurationError(
                    "Unknown condition(s): "
                    + ", ".join(sorted(unknown_conditions))
                )
            for filter_name in ("bet_exists", "bet_not_exists"):
                if filter_name in conditions:
                    cls._validate_bet_filter(
                        conditions[filter_name],
                        f"{action_id}.{filter_name}",
                    )
            operation = item.require("do")
            if not isinstance(operation, Mapping):
                raise ConfigurationError("Action 'do' must be a mapping.")
            operation = dict(operation)
            op_type = operation.get("type")
            if op_type not in cls.ACTION_TYPES:
                raise ConfigurationError(f"Unknown action type: {op_type}.")
            cls._validate_operation(operation, action_id)
            item.finish()
            actions.append(
                ActionSpec(action_id, trigger, dict(conditions), operation)
            )
        return tuple(actions)

    @classmethod
    def _validate_operation(
        cls,
        operation: Mapping[str, Any],
        action_id: str,
    ) -> None:
        """Validate one strategy operation and its operation-specific keys."""
        allowed: dict[str, set[str]] = {
            "place_bet": {
                "type",
                "kind",
                "amount",
                "number",
                "numbers",
                "hop_pair",
                "working",
            },
            "ensure_bet": {
                "type",
                "kind",
                "amount",
                "number",
                "numbers",
                "hop_pair",
                "working",
            },
            "take_odds": {
                "type",
                "parent_kind",
                "amount",
            },
            "remove_bets": {
                "type",
                "kind",
                "number",
                "numbers",
            },
            "regress_bets": {
                "type",
                "kind",
                "factor",
                "minimums",
            },
            "press_bet": {
                "type",
                "kind",
                "amount",
                "factor",
                "profit_fraction",
            },
            "set_working": {
                "type",
                "kind",
                "number",
                "numbers",
                "working",
            },
            "stop_session": {"type", "reason"},
            "message": {"type", "text"},
        }
        op_type = str(operation["type"])
        unknown = set(operation) - allowed[op_type]
        if unknown:
            raise ConfigurationError(
                f"Action {action_id} has unknown operation key(s): "
                + ", ".join(sorted(unknown))
            )
        if "kind" in operation:
            try:
                BetKind(str(operation["kind"]))
            except ValueError as exc:
                raise ConfigurationError(
                    f"Action {action_id} has an unknown wager kind."
                ) from exc
        if "parent_kind" in operation:
            parent_kind = BetKind(str(operation["parent_kind"]))
            if parent_kind not in {
                BetKind.PASS_LINE,
                BetKind.DONT_PASS,
                BetKind.COME,
                BetKind.DONT_COME,
            }:
                raise ConfigurationError(
                    f"Action {action_id} has an ineligible odds parent."
                )

    @staticmethod
    def _validate_bet_filter(raw: Any, path: str) -> None:
        """Validate a wager filter used by conditions or bulk actions."""
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{path} must be a mapping.")
        allowed = {"kind", "number", "numbers"}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigurationError(
                f"Unknown wager-filter key(s) at {path}: "
                + ", ".join(sorted(unknown))
            )
        if "kind" in raw:
            try:
                BetKind(str(raw["kind"]))
            except ValueError as exc:
                raise ConfigurationError(
                    f"Unknown wager kind at {path}."
                ) from exc
        if "number" in raw and "numbers" in raw:
            raise ConfigurationError(
                f"{path} cannot use both number and numbers."
            )

    @classmethod
    def _parse_scenarios(
        cls,
        raw: Any,
    ) -> tuple[ScenarioSpec, ...]:
        """Parse deterministic setup steps, rolls, and expected results."""
        if not isinstance(raw, list):
            raise ConfigurationError("scenarios must be a list.")
        scenarios: list[ScenarioSpec] = []
        for index, raw_scenario in enumerate(raw):
            item = StrictMapping(raw_scenario, f"scenarios[{index}]")
            name = str(item.require("name"))
            bankroll = as_money(item.require("bankroll"))
            setup = item.require("setup")
            rolls_raw = item.require("rolls")
            expected = item.require("expected")
            item.finish()
            if not isinstance(setup, list):
                raise ConfigurationError("Scenario setup must be a list.")
            if not isinstance(expected, Mapping):
                raise ConfigurationError("Scenario expected must be a map.")
            rolls: list[tuple[int, int]] = []
            for pair in rolls_raw:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ConfigurationError(
                        "Scenario rolls must be two-item lists."
                    )
                rolls.append((int(pair[0]), int(pair[1])))
            scenarios.append(
                ScenarioSpec(
                    name=name,
                    bankroll=bankroll,
                    setup=tuple(dict(action) for action in setup),
                    rolls=tuple(rolls),
                    expected=dict(expected),
                )
            )
        return tuple(scenarios)


# ---------------------------------------------------------------------------
# YAML STRATEGY CONTROLLER
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyContext:
    """Runtime values supplied while a strategy trigger is evaluated."""
    trigger: str
    event: Optional[RollEvent] = None
    settlement: Optional[Settlement] = None
    bet: Optional[Bet] = None


class YamlStrategyController:
    """Translate declarative YAML actions into controller method calls."""

    def __init__(
        self,
        controller: CrapsController,
        actions: Sequence[ActionSpec],
    ) -> None:
        """Bind validated actions to the table controller and output view."""
        self.controller = controller
        self.actions = tuple(actions)

    def fire(self, context: StrategyContext) -> None:
        """Evaluate and execute every action subscribed to one trigger."""
        for action in self.actions:
            if action.trigger != context.trigger:
                continue
            if not self._conditions_match(action.conditions, context):
                continue
            try:
                self._execute(action.operation, context)
            except (BetError, InsufficientFunds) as exc:
                self.controller.view.on_message(
                    f"Strategy action {action.action_id} skipped: {exc}"
                )

    def _conditions_match(
        self,
        conditions: Mapping[str, Any],
        context: StrategyContext,
    ) -> bool:
        """Check every condition against the current strategy context.

        Return True only when the complete condition mapping matches.
        """
        model = self.controller.model
        event = context.event
        settlement = context.settlement
        checks = {
            "phase": lambda value: model.phase.value == str(value),
            "point_in": lambda value: model.point in list(map(int, value)),
            "roll_total_in": lambda value: (
                event is not None
                and event.roll.total in list(map(int, value))
            ),
            "roll_number_gte": lambda value: model.roll_number >= int(value),
            "roll_number_lte": lambda value: model.roll_number <= int(value),
            "roll_every": lambda value: (
                model.roll_number > 0
                and model.roll_number % int(value) == 0
            ),
            "hand_number_gte": lambda value: model.hand_number >= int(value),
            "hand_number_lte": lambda value: model.hand_number <= int(value),
            "cash_gte": lambda value: model.ledger.cash >= as_money(value),
            "cash_lte": lambda value: model.ledger.cash <= as_money(value),
            "equity_gte": lambda value: model.equity >= as_money(value),
            "equity_lte": lambda value: model.equity <= as_money(value),
            "bet_exists": lambda value: self._matching_bets(value) != [],
            "bet_not_exists": lambda value: self._matching_bets(value) == [],
            "winning_kind": lambda value: (
                settlement is not None
                and settlement.outcome == Outcome.WIN
                and context.bet is not None
                and context.bet.kind == BetKind(str(value))
            ),
            "losing_kind": lambda value: (
                settlement is not None
                and settlement.outcome == Outcome.LOSS
                and context.bet is not None
                and context.bet.kind == BetKind(str(value))
            ),
        }
        return all(checks[key](value) for key, value in conditions.items())

    def _execute(
        self,
        operation: Mapping[str, Any],
        context: StrategyContext,
    ) -> None:
        """Dispatch one validated YAML operation to controller methods."""
        op_type = str(operation["type"])
        if op_type in {"place_bet", "ensure_bet"}:
            self._place_operation(operation, op_type == "ensure_bet")
        elif op_type == "take_odds":
            self._take_odds_operation(operation)
        elif op_type == "remove_bets":
            for bet in self._matching_bets(operation):
                self.controller.remove_bet(bet.bet_id)
        elif op_type == "regress_bets":
            minimums = {
                int(key): value
                for key, value in operation.get("minimums", {}).items()
            }
            self.controller.regress_bets(
                BetKind(str(operation["kind"])),
                operation["factor"],
                minimums,
            )
        elif op_type == "press_bet":
            self._press_operation(operation, context)
        elif op_type == "set_working":
            for bet in self._matching_bets(operation):
                self.controller.set_working(
                    bet.bet_id,
                    operation.get("working"),
                )
        elif op_type == "stop_session":
            self.controller.stop(str(operation["reason"]))
        elif op_type == "message":
            self.controller.view.on_message(str(operation["text"]))

    def _place_operation(
        self,
        operation: Mapping[str, Any],
        ensure: bool,
    ) -> None:
        """Place or ensure one or more wagers described by an operation."""
        method = (
            self.controller.ensure_bet
            if ensure
            else self.controller.place_bet
        )
        kind = BetKind(str(operation["kind"]))
        numbers = operation.get("numbers")
        if numbers is None:
            numbers = [operation.get("number")]
        for number in numbers:
            kwargs = {
                "kind": kind,
                "amount": operation["amount"],
                "number": None if number is None else int(number),
                "working": operation.get("working"),
            }
            if "hop_pair" in operation:
                kwargs["hop_pair"] = operation["hop_pair"]
            method(**kwargs)

    def _take_odds_operation(self, operation: Mapping[str, Any]) -> None:
        """Attach odds to every matching eligible parent contract."""
        parent_kind = BetKind(str(operation["parent_kind"]))
        for parent in self.controller.model.active_bets(parent_kind):
            if self.controller._parent_number(parent) is not None:
                self.controller.take_odds(
                    parent.bet_id,
                    operation.get("amount", "max"),
                )

    def _press_operation(
        self,
        operation: Mapping[str, Any],
        context: StrategyContext,
    ) -> None:
        """Press matching winning wagers using the configured sizing mode."""
        candidates = self._matching_bets(operation)
        if context.bet is not None:
            candidates = [
                bet
                for bet in candidates
                if bet.bet_id == context.bet.bet_id
            ]
        for bet in candidates:
            self.controller.press_bet(
                bet.bet_id,
                amount=operation.get("amount"),
                factor=operation.get("factor"),
                profit_fraction=operation.get("profit_fraction"),
                last_profit=(
                    context.settlement.profit
                    if context.settlement is not None
                    else None
                ),
            )

    def _matching_bets(self, spec: Mapping[str, Any]) -> list[Bet]:
        """Return active wagers selected by a validated filter mapping."""
        kind_raw = spec.get("kind")
        kind = BetKind(str(kind_raw)) if kind_raw is not None else None
        numbers = spec.get("numbers")
        if numbers is None and spec.get("number") is not None:
            numbers = [spec["number"]]
        bets = self.controller.model.active_bets(kind)
        if numbers is not None:
            allowed = set(map(int, numbers))
            bets = [bet for bet in bets if bet.number in allowed]
        return bets


# ---------------------------------------------------------------------------
# SESSION ORCHESTRATION, SCENARIOS, AND MONTE CARLO
# ---------------------------------------------------------------------------


class CrapsSession:
    """Run one bounded strategy session while preserving MVC boundaries."""

    def __init__(
        self,
        config: EngineConfig,
        dice: Optional[DiceSource] = None,
        view: Optional[CrapsView] = None,
    ) -> None:
        """Create a session with optional dice and view dependencies.

        Defaults are built from the validated engine configuration.
        """
        self.config = config
        ledger = BankrollLedger(config.session.bankroll)
        model = TableModel(ledger=ledger, rules=config.rules)
        self.controller = CrapsController(
            model,
            dice or RandomDice(config.session.seed),
            view or NullView(),
        )
        self.strategy = YamlStrategyController(
            self.controller,
            config.actions,
        )

    def run(self) -> SessionResult:
        """Run strategy events and dice rolls until a limit is reached.

        Return a SessionResult containing the final accounting state.
        """
        model = self.controller.model
        self.strategy.fire(StrategyContext("session_start"))
        self.strategy.fire(StrategyContext("hand_start"))
        while not model.session_stopped:
            reason = self._limit_reason()
            if reason:
                self.controller.stop(reason)
                break
            self.strategy.fire(StrategyContext("before_roll"))
            if model.phase == Phase.COME_OUT:
                self.strategy.fire(StrategyContext("come_out"))
            event = self.controller.roll_once()
            self.strategy.fire(StrategyContext("after_roll", event=event))
            for settlement in event.settlements:
                bet = model.bets[settlement.bet_id]
                if settlement.outcome == Outcome.WIN:
                    self.strategy.fire(
                        StrategyContext(
                            "bet_win",
                            event=event,
                            settlement=settlement,
                            bet=bet,
                        )
                    )
                elif settlement.outcome == Outcome.LOSS:
                    self.strategy.fire(
                        StrategyContext(
                            "bet_loss",
                            event=event,
                            settlement=settlement,
                            bet=bet,
                        )
                    )
            if event.point_established:
                self.strategy.fire(
                    StrategyContext("point_established", event=event)
                )
            if event.point_made:
                self.strategy.fire(StrategyContext("point_made", event=event))
            if event.seven_out:
                self.strategy.fire(StrategyContext("seven_out", event=event))
                if not model.session_stopped:
                    self.controller.start_new_hand()
                    self.strategy.fire(StrategyContext("hand_start"))
        result = SessionResult(
            opening_cash=model.ledger.opening_cash,
            closing_cash=model.ledger.cash,
            closing_equity=model.equity,
            rolls=model.roll_number,
            hands=model.hand_number,
            target_reached=model.equity >= self.config.session.target,
            stop_reason=model.stop_reason,
        )
        self.controller.view.on_session_end(result)
        return result

    def _limit_reason(self) -> str:
        """Return the first reached stop condition, or None to continue."""
        model = self.controller.model
        limits = self.config.session
        if model.equity >= limits.target:
            return "Target reached."
        if model.equity <= limits.stop_loss:
            return "Stop-loss reached."
        if model.roll_number >= limits.max_rolls:
            return "Maximum roll count reached."
        if model.hand_number > limits.max_hands:
            return "Maximum hand count reached."
        return ""


class ScenarioRunner:
    """Execute deterministic YAML scenarios as executable specifications."""

    def __init__(self, config: EngineConfig) -> None:
        """Store the configuration used by deterministic scenarios.

        Each scenario creates its own isolated controller and table model.
        """
        self.config = config

    def run_all(self) -> list[str]:
        """Run every configured scenario and return failure descriptions."""
        failures: list[str] = []
        for scenario in self.config.scenarios:
            try:
                self.run_one(scenario)
            except AssertionError as exc:
                failures.append(f"{scenario.name}: {exc}")
        return failures

    def run_one(self, scenario: ScenarioSpec) -> None:
        """Execute one scenario and validate its expected final state."""
        ledger = BankrollLedger(scenario.bankroll)
        model = TableModel(ledger=ledger, rules=self.config.rules)
        controller = CrapsController(
            model,
            ScriptedDice(scenario.rolls),
            NullView(),
        )
        for operation in scenario.setup:
            self._execute_setup(controller, operation)
        for index, _roll in enumerate(scenario.rolls):
            event = controller.roll_once()
            if event.seven_out and index < len(scenario.rolls) - 1:
                controller.start_new_hand()
        self._assert_expected(model, scenario.expected)

    @staticmethod
    def _execute_setup(
        controller: CrapsController,
        operation: Mapping[str, Any],
    ) -> None:
        """Apply one deterministic setup operation before scripted rolls."""
        op = dict(operation)
        op_type = op.pop("type")
        if op_type == "set_point":
            point = int(op.pop("point"))
            if point not in POINT_NUMBERS:
                raise ConfigurationError("Scenario point is invalid.")
            controller.model.point = point
        elif op_type == "set_bet_number":
            bet_index = int(op.pop("bet_index"))
            number = int(op.pop("number"))
            if number not in POINT_NUMBERS:
                raise ConfigurationError("Scenario bet number is invalid.")
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            active[bet_index].number = number
        elif op_type == "place_bet":
            kind = BetKind(str(op.pop("kind")))
            controller.place_bet(kind=kind, **op)
            op.clear()
        elif op_type == "take_odds":
            parent_index = int(op.pop("parent_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.take_odds(active[parent_index].bet_id, **op)
            op.clear()
        elif op_type == "set_working":
            bet_index = int(op.pop("bet_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.set_working(active[bet_index].bet_id, **op)
            op.clear()
        elif op_type == "increase_bet":
            bet_index = int(op.pop("bet_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.increase_bet(active[bet_index].bet_id, **op)
            op.clear()
        elif op_type == "reduce_bet":
            bet_index = int(op.pop("bet_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.reduce_bet(active[bet_index].bet_id, **op)
            op.clear()
        elif op_type == "remove_bet":
            bet_index = int(op.pop("bet_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.remove_bet(active[bet_index].bet_id)
        elif op_type == "regress_bets":
            kind = BetKind(str(op.pop("kind")))
            controller.regress_bets(kind=kind, **op)
            op.clear()
        elif op_type == "press_bet":
            bet_index = int(op.pop("bet_index"))
            active = sorted(
                controller.model.active_bets(),
                key=lambda bet: bet.bet_id,
            )
            controller.press_bet(active[bet_index].bet_id, **op)
            op.clear()
        else:
            raise ConfigurationError(
                f"Unknown scenario setup type: {op_type}."
            )
        if op:
            unknown = ", ".join(sorted(op))
            raise ConfigurationError(
                f"Unused scenario setup key(s): {unknown}."
            )

    @staticmethod
    def _assert_expected(
        model: TableModel,
        expected: Mapping[str, Any],
    ) -> None:
        """Compare the final model with every configured expectation."""
        allowed = {
            "cash",
            "equity",
            "point",
            "phase",
            "active_count",
            "active_by_kind",
            "hand_number",
            "roll_number",
        }
        unknown = set(expected) - allowed
        if unknown:
            raise ConfigurationError(
                "Unknown scenario expectation(s): "
                + ", ".join(sorted(unknown))
            )
        if "cash" in expected:
            assert model.ledger.cash == as_money(expected["cash"]), (
                f"cash {model.ledger.cash} != {expected['cash']}"
            )
        if "equity" in expected:
            assert model.equity == as_money(expected["equity"]), (
                f"equity {model.equity} != {expected['equity']}"
            )
        if "point" in expected:
            assert model.point == expected["point"], (
                f"point {model.point} != {expected['point']}"
            )
        if "phase" in expected:
            assert model.phase.value == expected["phase"]
        if "active_count" in expected:
            assert len(model.active_bets()) == int(expected["active_count"])
        if "active_by_kind" in expected:
            actual = Counter(bet.kind.value for bet in model.active_bets())
            wanted = Counter(
                {
                    str(key): int(value)
                    for key, value in expected["active_by_kind"].items()
                }
            )
            assert actual == wanted, f"active kinds {actual} != {wanted}"
        if "hand_number" in expected:
            assert model.hand_number == int(expected["hand_number"])
        if "roll_number" in expected:
            assert model.roll_number == int(expected["roll_number"])


class MonteCarloRunner:
    """Repeat complete sessions and summarize outcomes without shared state."""

    def __init__(self, config: EngineConfig) -> None:
        """Store the immutable configuration shared by all simulated trials."""
        self.config = config

    def run(self, trials: int) -> dict[str, Any]:
        """Run complete sessions and aggregate their results.

        Report target hits, net outcomes, and average roll counts.
        """
        if trials <= 0:
            raise ValueError("trials must be positive.")
        results: list[SessionResult] = []
        base_seed = self.config.session.seed
        for index in range(trials):
            session_raw = copy.deepcopy(self.config)
            seed = None if base_seed is None else base_seed + index
            session_config = SessionConfig(
                bankroll=session_raw.session.bankroll,
                target=session_raw.session.target,
                stop_loss=session_raw.session.stop_loss,
                max_rolls=session_raw.session.max_rolls,
                max_hands=session_raw.session.max_hands,
                seed=seed,
            )
            trial_config = EngineConfig(
                version=session_raw.version,
                name=session_raw.name,
                description=session_raw.description,
                session=session_config,
                rules=session_raw.rules,
                actions=session_raw.actions,
                scenarios=session_raw.scenarios,
            )
            results.append(CrapsSession(trial_config).run())
        nets = [float(result.net) for result in results]
        return {
            "trials": trials,
            "target_hits": sum(result.target_reached for result in results),
            "target_rate": sum(
                result.target_reached for result in results
            )
            / trials,
            "mean_net": statistics.fmean(nets),
            "median_net": statistics.median(nets),
            "mean_rolls": statistics.fmean(
                result.rolls for result in results
            ),
        }


# ---------------------------------------------------------------------------
# EMBEDDED UNIT TESTS
# ---------------------------------------------------------------------------


class EngineTestCase(unittest.TestCase):
    """Shared deterministic table factory for settlement tests."""

    def make_controller(
        self,
        rolls: Sequence[Sequence[int]],
        bankroll: Any = 1000,
        rules: Optional[TableRules] = None,
    ) -> CrapsController:
        """Build a funded controller with optional rules and scripted dice."""
        chosen_rules = rules or TableRules(
            enforce_increments=False,
            payout_rounding="exact",
        )
        model = TableModel(
            ledger=BankrollLedger(as_money(bankroll)),
            rules=chosen_rules,
        )
        return CrapsController(model, ScriptedDice(rolls), NullView())


class TestDiceAndArchitecture(EngineTestCase):
    """Tests for dice sources, value objects, and MVC separation."""
    def test_dice_roll_properties(self) -> None:
        """Verify sum, pair, and unordered-pair properties of a dice roll."""
        roll = DiceRoll(3, 4)
        self.assertEqual(roll.total, 7)
        self.assertFalse(roll.is_pair)
        self.assertEqual(roll.unordered_pair, (3, 4))

    def test_invalid_die_is_rejected(self) -> None:
        """Verify die faces outside one through six are rejected."""
        with self.assertRaises(ValueError):
            DiceRoll(0, 6)

    def test_mvc_types_are_separate(self) -> None:
        """Verify model, view, and controller remain distinct object types."""
        controller = self.make_controller([(3, 4)])
        self.assertIsInstance(controller.model, TableModel)
        self.assertIsInstance(controller.view, CrapsView)
        self.assertIsInstance(controller, CrapsController)

    def test_random_dice_are_seed_reproducible(self) -> None:
        """Verify equal integer seeds produce equal dice sequences."""
        first = RandomDice(99)
        second = RandomDice(99)
        self.assertEqual(
            [first.roll() for _ in range(100)],
            [second.roll() for _ in range(100)],
        )

    def test_random_dice_cover_all_faces_without_extreme_bias(self) -> None:
        """Check a seeded sample covers every face without severe imbalance."""
        dice = RandomDice(20260709)
        faces = Counter()
        for _ in range(36000):
            roll = dice.roll()
            faces[roll.die_1] += 1
            faces[roll.die_2] += 1
        expected = 12000
        for face in range(1, 7):
            self.assertLess(abs(faces[face] - expected), 500)


class TestLineAndComeBets(EngineTestCase):
    """Tests for line, Come, and Don’t Come contract behavior."""
    def test_pass_line_natural_and_craps(self) -> None:
        """Verify Pass Line wins on naturals and loses on come-out craps."""
        win = self.make_controller([(3, 4)], bankroll=100)
        win.place_bet(BetKind.PASS_LINE, 10)
        win.roll_once()
        self.assertEqual(win.model.ledger.cash, Decimal("110"))
        lose = self.make_controller([(1, 1)], bankroll=100)
        lose.place_bet(BetKind.PASS_LINE, 10)
        lose.roll_once()
        self.assertEqual(lose.model.ledger.cash, Decimal("90"))

    def test_dont_pass_bar_push_keeps_stake_active(self) -> None:
        """Verify the Don’t Pass bar number pushes.

        The contract stake must remain active after the standoff.
        """
        controller = self.make_controller([(6, 6)], bankroll=100)
        bet = controller.place_bet(BetKind.DONT_PASS, 10)
        event = controller.roll_once()
        self.assertEqual(event.settlements[0].outcome, Outcome.PUSH)
        self.assertTrue(bet.is_active)
        self.assertEqual(controller.model.equity, Decimal("100"))

    def test_bar_two_variant_makes_twelve_a_dont_winner(self) -> None:
        """Verify a Bar-2 table makes twelve a Don’t Pass winner."""
        rules = TableRules(
            dont_bar=2,
            enforce_increments=False,
            payout_rounding="exact",
        )
        push = self.make_controller([(1, 1)], bankroll=100, rules=rules)
        push_bet = push.place_bet(BetKind.DONT_PASS, 10)
        push.roll_once()
        self.assertTrue(push_bet.is_active)
        self.assertEqual(push.model.equity, Decimal("100"))
        win = self.make_controller([(6, 6)], bankroll=100, rules=rules)
        win.place_bet(BetKind.DONT_PASS, 10)
        win.roll_once()
        self.assertEqual(win.model.ledger.cash, Decimal("110"))

    def test_point_cycle_and_seven_out(self) -> None:
        """Verify point establishment and point resolution.

        Confirm that a seven-out returns the table to come-out state.
        """
        controller = self.make_controller([(2, 4), (3, 4)], bankroll=100)
        controller.place_bet(BetKind.PASS_LINE, 10)
        first = controller.roll_once()
        self.assertTrue(first.point_established)
        self.assertEqual(controller.model.point, 6)
        second = controller.roll_once()
        self.assertTrue(second.seven_out)
        self.assertEqual(controller.model.point, None)
        self.assertEqual(controller.model.ledger.cash, Decimal("90"))

    def test_come_travels_and_wins(self) -> None:
        """Verify a Come wager travels and wins when its number repeats."""
        controller = self.make_controller([(2, 3), (1, 4)], bankroll=100)
        controller.model.point = 8
        bet = controller.place_bet(BetKind.COME, 10)
        controller.roll_once()
        self.assertEqual(bet.number, 5)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("110"))
        self.assertFalse(bet.is_active)

    def test_dont_come_travels_and_wins_on_seven(self) -> None:
        """Verify a Don’t Come wager travels and wins when seven appears."""
        controller = self.make_controller([(3, 3), (3, 4)], bankroll=100)
        controller.model.point = 8
        bet = controller.place_bet(BetKind.DONT_COME, 10)
        controller.roll_once()
        self.assertEqual(bet.number, 6)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("110"))


class TestOdds(EngineTestCase):
    """Tests for taking and laying odds on contract wagers."""
    def test_pass_odds_pay_true_odds(self) -> None:
        """Verify Pass Line odds pay the correct true-odds profit."""
        controller = self.make_controller([(2, 2)], bankroll=100)
        parent = controller.place_bet(BetKind.PASS_LINE, 10)
        controller.model.point = 4
        odds = controller.take_odds(parent.bet_id, 20)
        controller.roll_once()
        self.assertFalse(parent.is_active)
        self.assertFalse(odds.is_active)
        self.assertEqual(controller.model.ledger.cash, Decimal("150"))

    def test_dont_pass_lay_odds_pay_true_odds(self) -> None:
        """Verify Don’t Pass lay odds pay the correct true-odds profit."""
        controller = self.make_controller([(3, 4)], bankroll=100)
        parent = controller.place_bet(BetKind.DONT_PASS, 10)
        controller.model.point = 4
        controller.take_odds(parent.bet_id, 20)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("120"))

    def test_come_odds_off_are_returned_when_flat_loses(self) -> None:
        """Verify off Come odds are returned when the flat wager resolves."""
        rules = TableRules(
            enforce_increments=False,
            payout_rounding="exact",
            come_odds_working_on_comeout=False,
        )
        controller = self.make_controller([(3, 4)], bankroll=100, rules=rules)
        controller.model.point = 8
        come = controller.place_bet(BetKind.COME, 10)
        come.number = 5
        controller.model.point = None
        controller.take_odds(come.bet_id, 10)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("90"))
        self.assertEqual(len(controller.model.active_bets()), 0)

    def test_max_odds_limits_are_enforced(self) -> None:
        """Verify configured maximum odds cannot be exceeded."""
        controller = self.make_controller([], bankroll=1000)
        parent = controller.place_bet(BetKind.PASS_LINE, 10)
        controller.model.point = 4
        with self.assertRaises(BetError):
            controller.take_odds(parent.bet_id, 31)


    def test_direct_odds_cannot_bypass_limit_or_direction(self) -> None:
        """Verify direct odds placement cannot bypass limits or direction."""
        controller = self.make_controller([], bankroll=1000)
        parent = controller.place_bet(BetKind.PASS_LINE, 10)
        controller.model.point = 4
        with self.assertRaises(BetError):
            controller.place_bet(
                BetKind.TAKE_ODDS,
                31,
                number=4,
                parent_id=parent.bet_id,
            )
        with self.assertRaises(BetError):
            controller.place_bet(
                BetKind.LAY_ODDS,
                10,
                number=4,
                parent_id=parent.bet_id,
            )


class TestPersistentAndPropositionBets(EngineTestCase):
    """Tests for persistent wagers and one-roll propositions."""
    def test_place_6_wins_and_remains(self) -> None:
        """Verify a winning Place 6 credits profit and remains active."""
        controller = self.make_controller([(3, 3)], bankroll=100)
        controller.model.point = 5
        bet = controller.place_bet(BetKind.PLACE, 12, number=6)
        controller.roll_once()
        self.assertTrue(bet.is_active)
        self.assertEqual(controller.model.ledger.cash, Decimal("102"))
        self.assertEqual(controller.model.equity, Decimal("114"))

    def test_place_bet_is_off_on_comeout_by_default(self) -> None:
        """Verify a Place wager does not resolve on come-out by default."""
        controller = self.make_controller([(3, 4)], bankroll=100)
        bet = controller.place_bet(BetKind.PLACE, 12, number=6)
        controller.roll_once()
        self.assertTrue(bet.is_active)
        self.assertEqual(controller.model.equity, Decimal("100"))

    def test_hardway_wins_hard_and_loses_easy(self) -> None:
        """Verify Hardways win as pairs and lose to easy combinations."""
        controller = self.make_controller([(3, 3), (1, 5)], bankroll=100)
        controller.model.point = 5
        bet = controller.place_bet(BetKind.HARDWAY, 1, number=6)
        controller.roll_once()
        self.assertTrue(bet.is_active)
        self.assertEqual(controller.model.equity, Decimal("109"))
        controller.roll_once()
        self.assertFalse(bet.is_active)
        self.assertEqual(controller.model.ledger.cash, Decimal("108"))

    def test_big_6_loses_to_seven(self) -> None:
        """Verify Big 6 loses and comes down when seven rolls."""
        controller = self.make_controller([(3, 4)], bankroll=100)
        controller.model.point = 5
        bet = controller.place_bet(BetKind.BIG_6, 5)
        controller.roll_once()
        self.assertFalse(bet.is_active)
        self.assertEqual(controller.model.ledger.cash, Decimal("95"))

    def test_field_double_and_triple_rules(self) -> None:
        """Verify configurable double-two and triple-twelve Field payouts."""
        two = self.make_controller([(1, 1)], bankroll=100)
        two.place_bet(BetKind.FIELD, 10)
        two.roll_once()
        self.assertEqual(two.model.ledger.cash, Decimal("120"))
        twelve = self.make_controller([(6, 6)], bankroll=100)
        twelve.place_bet(BetKind.FIELD, 10)
        twelve.roll_once()
        self.assertEqual(twelve.model.ledger.cash, Decimal("130"))

    def test_one_roll_loss_is_removed(self) -> None:
        """Verify a losing one-roll proposition is removed immediately."""
        controller = self.make_controller([(2, 4)], bankroll=100)
        bet = controller.place_bet(BetKind.ANY_SEVEN, 5)
        controller.roll_once()
        self.assertFalse(bet.is_active)
        self.assertEqual(controller.model.ledger.cash, Decimal("95"))

    def test_horn_math_uses_quarter_stakes(self) -> None:
        """Verify Horn settlement divides the stake among four components."""
        controller = self.make_controller([(1, 1)], bankroll=100)
        controller.place_bet(BetKind.HORN, 4)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("127"))

    def test_world_seven_is_a_push_at_four_to_one(self) -> None:
        """Verify a five-unit World wager pushes on seven.

        One unit wins at four to one while the other four units lose.
        """
        controller = self.make_controller([(3, 4)], bankroll=100)
        controller.place_bet(BetKind.WORLD, 5)
        controller.roll_once()
        self.assertEqual(controller.model.ledger.cash, Decimal("100"))

    def test_hop_easy_and_hard_payouts(self) -> None:
        """Verify easy and hard Hop combinations use their correct payouts."""
        easy = self.make_controller([(2, 5)], bankroll=100)
        easy.place_bet(BetKind.HOP, 1, hop_pair=(5, 2))
        easy.roll_once()
        self.assertEqual(easy.model.ledger.cash, Decimal("115"))
        hard = self.make_controller([(4, 4)], bankroll=100)
        hard.place_bet(BetKind.HOP, 1, hop_pair=(4, 4))
        hard.roll_once()
        self.assertEqual(hard.model.ledger.cash, Decimal("130"))


class TestVigAndBetManagement(EngineTestCase):
    """Tests for commissions and wager-management operations."""
    def test_buy_vig_on_win(self) -> None:
        """Verify Buy commission is deducted when configured on a win."""
        rules = TableRules(
            enforce_increments=False,
            payout_rounding="exact",
            buy_vig_timing="on_win",
            buy_vig_basis="wager",
        )
        controller = self.make_controller([(2, 2)], bankroll=100, rules=rules)
        controller.model.point = 5
        controller.place_bet(BetKind.BUY, 20, number=4)
        controller.roll_once()
        self.assertEqual(controller.model.equity, Decimal("139"))

    def test_lay_vig_upfront(self) -> None:
        """Verify Lay commission is charged when configured upfront."""
        rules = TableRules(
            enforce_increments=False,
            payout_rounding="exact",
            lay_vig_timing="upfront",
            lay_vig_basis="win",
        )
        controller = self.make_controller([(3, 4)], bankroll=100, rules=rules)
        controller.model.point = 5
        controller.place_bet(BetKind.LAY, 20, number=4)
        self.assertEqual(controller.model.ledger.cash, Decimal("79.5"))
        controller.roll_once()
        self.assertEqual(controller.model.equity, Decimal("109.5"))

    def test_increasing_upfront_vig_bet_charges_added_vig(self) -> None:
        """Verify increasing an upfront-vig wager charges only added vig."""
        rules = TableRules(
            enforce_increments=False,
            payout_rounding="exact",
            lay_vig_timing="upfront",
            lay_vig_basis="win",
        )
        controller = self.make_controller([], bankroll=100, rules=rules)
        bet = controller.place_bet(BetKind.LAY, 20, number=4)
        self.assertEqual(controller.model.ledger.cash, Decimal("79.5"))
        controller.increase_bet(bet.bet_id, 40)
        self.assertEqual(controller.model.ledger.cash, Decimal("59"))
        self.assertEqual(controller.model.equity, Decimal("99"))

    def test_remove_reduce_increase_preserve_equity(self) -> None:
        """Verify wager-management operations maintain correct equity."""
        controller = self.make_controller([], bankroll=100)
        bet = controller.place_bet(BetKind.PLACE, 20, number=5)
        self.assertEqual(controller.model.equity, Decimal("100"))
        controller.reduce_bet(bet.bet_id, 10)
        self.assertEqual(controller.model.equity, Decimal("100"))
        controller.increase_bet(bet.bet_id, 15)
        self.assertEqual(controller.model.equity, Decimal("100"))
        controller.remove_bet(bet.bet_id)
        self.assertEqual(controller.model.ledger.cash, Decimal("100"))

    def test_regress_returns_removed_stake(self) -> None:
        """Verify regression returns every removed chip to rack cash."""
        controller = self.make_controller([], bankroll=100)
        controller.place_bet(BetKind.PLACE, 20, number=5)
        controller.regress_bets(BetKind.PLACE, Decimal("0.5"))
        active = controller.model.active_bets(BetKind.PLACE)[0]
        self.assertEqual(active.amount, Decimal("10"))
        self.assertEqual(controller.model.ledger.cash, Decimal("90"))

    def test_press_debits_only_added_stake(self) -> None:
        """Verify a press reserves only the added stake.

        Existing wagered funds must not be debited a second time.
        """
        controller = self.make_controller([], bankroll=100)
        bet = controller.place_bet(BetKind.PLACE, 10, number=5)
        controller.press_bet(bet.bet_id, amount=5)
        self.assertEqual(bet.amount, Decimal("15"))
        self.assertEqual(controller.model.ledger.cash, Decimal("85"))
        self.assertEqual(controller.model.equity, Decimal("100"))


class TestExhaustiveOneRollMath(EngineTestCase):
    """Exhaustive expectation tests over all 36 dice combinations."""
    def expected_net(self, kind: BetKind, amount: Any) -> Money:
        """Calculate exact expected net over all 36 ordered dice outcomes."""
        total = Decimal("0")
        for die_1 in range(1, 7):
            for die_2 in range(1, 7):
                controller = self.make_controller(
                    [(die_1, die_2)],
                    bankroll=100,
                )
                controller.place_bet(kind, amount)
                controller.roll_once()
                total += controller.model.equity - Decimal("100")
        return total / Decimal("36")

    def test_any_seven_exact_house_edge(self) -> None:
        """Verify Any Seven expectation over all ordered dice outcomes."""
        expected = self.expected_net(BetKind.ANY_SEVEN, 1)
        self.assertEqual(expected, Decimal(-1) / Decimal(6))

    def test_any_craps_exact_house_edge(self) -> None:
        """Verify Any Craps expectation over all ordered dice outcomes."""
        expected = self.expected_net(BetKind.ANY_CRAPS, 1)
        self.assertEqual(expected, Decimal(-1) / Decimal(9))

    def test_field_exact_expectation_for_double_two_triple_twelve(
        self,
    ) -> None:
        """Verify Field expectation with double two and triple twelve."""
        expected = self.expected_net(BetKind.FIELD, 1)
        self.assertEqual(expected, Decimal(-1) / Decimal(36))


class TestYamlAndScenarios(EngineTestCase):
    """Tests for strict YAML parsing and end-to-end scenario execution."""
    @classmethod
    def yaml_path(cls) -> Path:
        """Return the showcase YAML path used by integration tests.

        The path is resolved beside the single-file engine source.
        """
        return Path(__file__).with_name("uce_strategy_full_showcase.yaml")

    def test_companion_yaml_loads_strictly(self) -> None:
        """Verify the showcase YAML parses under schema version 1.

        Strict parsing must accept every documented field in the file.
        """
        config = YamlConfigLoader.load(self.yaml_path())
        self.assertGreater(len(config.actions), 0)
        self.assertGreater(len(config.scenarios), 0)

    def test_showcase_covers_entire_yaml_action_surface(self) -> None:
        """Verify the showcase exercises every supported strategy action."""
        config = YamlConfigLoader.load(self.yaml_path())
        triggers = {action.trigger for action in config.actions}
        operations = {
            str(action.operation["type"]) for action in config.actions
        }
        conditions: set[str] = set()
        for action in config.actions:
            conditions.update(action.conditions)
        self.assertEqual(triggers, YamlConfigLoader.ACTION_TRIGGERS)
        self.assertEqual(operations, YamlConfigLoader.ACTION_TYPES)
        self.assertEqual(conditions, YamlConfigLoader.CONDITION_KEYS)

    def test_unknown_nested_bet_filter_is_rejected(self) -> None:
        """Verify unknown nested wager-filter keys fail validation."""
        with self.yaml_path().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["strategy"]["actions"][4]["when"]["bet_not_exists"][
            "mystery"
        ] = True
        with self.assertRaises(ConfigurationError):
            YamlConfigLoader.from_mapping(raw)

    def test_source_obeys_basic_pep8_whitespace_limits(self) -> None:
        """Verify source lines and trailing whitespace meet project limits."""
        source_lines = Path(__file__).read_text(encoding="utf-8").splitlines()
        self.assertFalse(any("\t" in line for line in source_lines))
        self.assertFalse(any(line.rstrip() != line for line in source_lines))
        self.assertLessEqual(max(map(len, source_lines)), 79)

    def test_monte_carlo_returns_numeric_aggregate(self) -> None:
        """Verify Monte Carlo output contains numeric aggregate statistics."""
        config = YamlConfigLoader.load(self.yaml_path())
        summary = MonteCarloRunner(config).run(3)
        self.assertEqual(summary["trials"], 3)
        self.assertGreaterEqual(summary["target_rate"], 0)
        self.assertLessEqual(summary["target_rate"], 1)

    def test_monte_carlo_preserves_null_seed(self) -> None:
        """Verify null and integer Monte Carlo seed behavior.

        Null stays nondeterministic; integer seeds increment by trial.
        """
        config = YamlConfigLoader.load(self.yaml_path())
        null_config = replace(
            config,
            session=replace(config.session, seed=None),
        )
        seeded_config = replace(
            config,
            session=replace(config.session, seed=100),
        )
        result = SessionResult(
            opening_cash=Decimal("100"),
            closing_cash=Decimal("100"),
            closing_equity=Decimal("100"),
            rolls=1,
            hands=1,
            target_reached=False,
            stop_reason="test",
        )
        module = sys.modules[__name__]
        with mock.patch.object(module, "CrapsSession") as session_class:
            session_class.return_value.run.return_value = result
            MonteCarloRunner(null_config).run(3)
            null_seeds = [
                call.args[0].session.seed
                for call in session_class.call_args_list
            ]
        self.assertEqual(null_seeds, [None, None, None])
        with mock.patch.object(module, "CrapsSession") as session_class:
            session_class.return_value.run.return_value = result
            MonteCarloRunner(seeded_config).run(3)
            integer_seeds = [
                call.args[0].session.seed
                for call in session_class.call_args_list
            ]
        self.assertEqual(integer_seeds, [100, 101, 102])

    def test_unknown_root_key_is_rejected(self) -> None:
        """Verify unknown top-level YAML keys fail validation."""
        with self.yaml_path().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["mystery"] = True
        with self.assertRaises(ConfigurationError):
            YamlConfigLoader.from_mapping(raw)

    def test_all_yaml_scenarios_pass(self) -> None:
        """Verify every deterministic showcase scenario succeeds."""
        config = YamlConfigLoader.load(self.yaml_path())
        failures = ScenarioRunner(config).run_all()
        self.assertEqual(failures, [])

    def test_showcase_strategy_runs_to_a_defined_stop(self) -> None:
        """Verify the showcase session reaches a recognized stop condition."""
        config = YamlConfigLoader.load(self.yaml_path())
        result = CrapsSession(config).run()
        self.assertGreater(result.rolls, 0)
        self.assertTrue(result.stop_reason)


# ---------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser and its mutually exclusive run modes."""
    parser = argparse.ArgumentParser(
        description="Universal, YAML-driven craps simulation engine."
    )
    parser.add_argument(
        "strategy",
        nargs="?",
        type=Path,
        help="Path to a version-1 craps YAML configuration.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--simulate",
        action="store_true",
        help="Run one silent configured session and print the summary.",
    )
    mode.add_argument(
        "--audit",
        action="store_true",
        help="Run one configured session with roll-by-roll output.",
    )
    mode.add_argument(
        "--run-scenarios",
        action="store_true",
        help="Execute all deterministic scenarios in the YAML file.",
    )
    mode.add_argument(
        "--monte-carlo",
        type=int,
        metavar="TRIALS",
        help="Run multiple complete sessions and aggregate results.",
    )
    mode.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the embedded unittest suite.",
    )
    return parser


def run_tests() -> int:
    """Run the embedded unittest suite and return a process exit status."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and execute the selected engine mode."""
    args = build_parser().parse_args(argv)
    if args.run_tests:
        return run_tests()
    if args.strategy is None:
        raise SystemExit("A YAML strategy path is required for this mode.")
    config = YamlConfigLoader.load(args.strategy)
    if args.run_scenarios:
        failures = ScenarioRunner(config).run_all()
        if failures:
            print("Scenario failures:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print(f"All {len(config.scenarios)} YAML scenarios passed.")
        return 0
    if args.monte_carlo:
        summary = MonteCarloRunner(config).run(args.monte_carlo)
        print(f"Trials:       {summary['trials']}")
        print(f"Target hits:  {summary['target_hits']}")
        print(f"Target rate:  {summary['target_rate']:.2%}")
        print(f"Mean net:     ${summary['mean_net']:,.2f}")
        print(f"Median net:   ${summary['median_net']:,.2f}")
        print(f"Mean rolls:   {summary['mean_rolls']:,.2f}")
        return 0
    view: CrapsView = ConsoleView()
    if args.simulate:
        view = NullView()
    result = CrapsSession(config, view=view).run()
    if args.simulate:
        ConsoleView().on_session_end(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
