"""FP-Growth Association Rule Mining for TechNova AI Recommendation Engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class FPNode:
    item: str | None
    count: int
    parent: FPNode | None
    children: dict[str, FPNode] = field(default_factory=dict)
    node_link: FPNode | None = None


from ..domain.models import AssociationRule


class FPTree:
    def __init__(self) -> None:
        self.root = FPNode(item=None, count=0, parent=None)
        self.header_table: dict[str, list[Any]] = {}  # item -> [count, head_node]

    def add_transaction(self, items: list[str], count: int = 1) -> None:
        current_node = self.root
        for item in items:
            if item in current_node.children:
                current_node.children[item].count += count
            else:
                new_node = FPNode(item=item, count=count, parent=current_node)
                current_node.children[item] = new_node

                # Link into header table
                if item in self.header_table:
                    # Append to end of linked list
                    head = self.header_table[item][1]
                    if head is None:
                        self.header_table[item][1] = new_node
                    else:
                        tail = head
                        while tail.node_link is not None:
                            tail = tail.node_link
                        tail.node_link = new_node
            current_node = current_node.children[item]


def _build_fp_tree(
    transactions: list[list[str]],
    min_count: int,
) -> tuple[FPTree | None, dict[str, int]]:
    # Step 1: Count item frequencies
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1

    # Filter by minimum count
    frequent_items = {item: cnt for item, cnt in item_counts.items() if cnt >= min_count}
    if not frequent_items:
        return None, {}

    # Sort order: descending frequency, then alphabetical
    tree = FPTree()
    for item, cnt in frequent_items.items():
        tree.header_table[item] = [cnt, None]

    # Step 2: Insert transactions into tree
    for txn in transactions:
        # Keep only frequent items, sort by descending frequency
        filtered_sorted = [
            item for item in txn if item in frequent_items
        ]
        filtered_sorted.sort(key=lambda it: (-frequent_items[it], it))
        if filtered_sorted:
            tree.add_transaction(filtered_sorted)

    return tree, frequent_items


def _mine_fp_tree(
    tree: FPTree,
    frequent_items: dict[str, int],
    min_count: int,
    prefix: set[str],
    frequent_itemsets: dict[frozenset[str], int],
) -> None:
    # Sort items in header table by ascending frequency
    sorted_items = sorted(tree.header_table.keys(), key=lambda it: frequent_items[it])

    for item in sorted_items:
        new_prefix = prefix.copy()
        new_prefix.add(item)
        item_count = tree.header_table[item][0]
        frequent_itemsets[frozenset(new_prefix)] = item_count

        # Build conditional pattern base
        conditional_patterns: list[list[str]] = []
        node = tree.header_table[item][1]
        while node is not None:
            # Ascend to root
            path: list[str] = []
            curr = node.parent
            while curr is not None and curr.item is not None:
                path.append(curr.item)
                curr = curr.parent
            if path:
                # Add path multiplied by node.count
                for _ in range(node.count):
                    conditional_patterns.append(path)
            node = node.node_link

        # Build conditional FP-tree
        cond_tree, cond_items = _build_fp_tree(conditional_patterns, min_count)
        if cond_tree is not None and cond_tree.header_table:
            _mine_fp_tree(cond_tree, cond_items, min_count, new_prefix, frequent_itemsets)


class FPGrowthModel:
    """Frequent Pattern Growth (FP-Growth) association rule recommender."""

    def __init__(
        self,
        min_support: float = 0.005,
        min_confidence: float = 0.15,
        min_lift: float = 1.0,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.total_transactions = 0
        self.item_support: dict[str, float] = {}
        self.frequent_itemsets: dict[frozenset[str], int] = {}
        self.rules: list[AssociationRule] = []
        # Fast rule index: antecedent -> list of (consequent_item, rule)
        self.rule_index: dict[frozenset[str], list[tuple[str, AssociationRule]]] = defaultdict(list)

    def fit(
        self,
        df_baskets: pd.DataFrame,
        min_support: float | None = None,
        min_confidence: float | None = None,
        min_lift: float | None = None,
    ) -> FPGrowthModel:
        """Trains FP-Growth model on line-item transaction basket DataFrame."""
        if min_support is not None:
            self.min_support = min_support
        if min_confidence is not None:
            self.min_confidence = min_confidence
        if min_lift is not None:
            self.min_lift = min_lift

        # Group by transaction_id to build basket itemsets
        baskets_grouped = (
            df_baskets.groupby("transaction_id")["product_id"]
            .apply(lambda s: list(set(s)))
            .tolist()
        )
        self.total_transactions = len(baskets_grouped)
        if self.total_transactions == 0:
            return self

        min_count = max(1, int(self.min_support * self.total_transactions))

        # Build and mine FP-Tree
        tree, frequent_items = _build_fp_tree(baskets_grouped, min_count)
        self.frequent_itemsets = {}
        if tree is not None:
            _mine_fp_tree(tree, frequent_items, min_count, set(), self.frequent_itemsets)

        # Store single-item support
        for item, count in frequent_items.items():
            self.item_support[item] = count / self.total_transactions

        # Generate association rules
        self.rules = []
        self.rule_index.clear()

        for itemset, itemset_count in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            support_ab = itemset_count / self.total_transactions

            # Generate all non-empty antecedent subsets
            for k in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, k):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    if not consequent:
                        continue

                    # Lookup antecedent count
                    antecedent_count = self.frequent_itemsets.get(antecedent, 0)
                    if antecedent_count == 0:
                        continue

                    confidence = itemset_count / antecedent_count
                    if confidence < self.min_confidence:
                        continue

                    # Calculate lift
                    consequent_count = self.frequent_itemsets.get(consequent, 0)
                    support_b = consequent_count / self.total_transactions
                    if support_b <= 0:
                        continue

                    lift = confidence / support_b
                    if lift < self.min_lift:
                        continue

                    rule = AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=support_ab,
                        confidence=confidence,
                        lift=lift,
                        count=itemset_count,
                    )
                    self.rules.append(rule)

                    # Index rules for single-item consequents
                    if len(consequent) == 1:
                        conseq_item = next(iter(consequent))
                        self.rule_index[antecedent].append((conseq_item, rule))

        # Sort rules by confidence descending
        self.rules.sort(key=lambda r: (-r.confidence, -r.lift))
        return self

    def recommend(
        self,
        item_ids: list[str],
        top_k: int = 10,
        exclude_input: bool = True,
    ) -> list[tuple[str, float]]:
        """Recommends complementary products given one or more antecedent product IDs.

        Returns list of (product_id, association_confidence_score).
        """
        input_set = set(item_ids)
        candidate_scores: dict[str, float] = defaultdict(float)

        # Search exact match and subset matches
        for antecedent, rule_list in self.rule_index.items():
            if antecedent.issubset(input_set):
                for target_item, rule in rule_list:
                    if exclude_input and target_item in input_set:
                        continue
                    # Weight score by confidence and normalized log-lift
                    score = float(rule.confidence)
                    if score > candidate_scores[target_item]:
                        candidate_scores[target_item] = score

        # Rank candidates
        ranked = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        """Serializes trained model to joblib file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> FPGrowthModel:
        """Loads trained model from joblib file."""
        return joblib.load(path)
