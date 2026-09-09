"""
Global (not per-subscriber) classification caches -- an interest's
category mapping, retrieval-query expansion, and a category's keyness
scores. Same interest text means the same cached value no matter which
subscriber set it.
"""

from sqlalchemy import text


class InterestCacheMixin:
    def get_cached_interest_categories(self, interests: list[str]) -> list[tuple]:
        if not interests:
            return []
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":i{n}" for n in range(len(interests)))
            params = {f"i{n}": v for n, v in enumerate(interests)}
            return conn.execute(text(
                f"SELECT interest, categories FROM interest_categories WHERE interest IN ({placeholders})"
            ), params).fetchall()

    def set_interest_categories(self, interest: str, categories_json: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(
                """
                INSERT INTO interest_categories (interest, categories) VALUES (:interest, :categories)
                ON CONFLICT(interest) DO UPDATE SET categories = excluded.categories
                """
            ), {"interest": interest, "categories": categories_json})

    def get_interest_query_expansion(self, interest: str) -> str | None:
        with self._engine.begin() as conn:
            row = conn.execute(
                text("SELECT expansion FROM interest_query_expansions WHERE interest = :interest"),
                {"interest": interest},
            ).fetchone()
        return row[0] if row else None

    def set_interest_query_expansion(self, interest: str, expansion: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(
                """
                INSERT INTO interest_query_expansions (interest, expansion) VALUES (:interest, :expansion)
                ON CONFLICT(interest) DO UPDATE SET expansion = excluded.expansion
                """
            ), {"interest": interest, "expansion": expansion})

    def get_subscriber_interest_definition(self, chat_id: int, interest: str) -> str | None:
        with self._engine.begin() as conn:
            row = conn.execute(
                text("SELECT expansion FROM subscriber_interest_definitions "
                     "WHERE chat_id = :chat_id AND interest = :interest"),
                {"chat_id": chat_id, "interest": interest},
            ).fetchone()
        return row[0] if row else None

    def set_subscriber_interest_definition(self, chat_id: int, interest: str, expansion: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(
                """
                INSERT INTO subscriber_interest_definitions (chat_id, interest, expansion)
                VALUES (:chat_id, :interest, :expansion)
                ON CONFLICT(chat_id, interest) DO UPDATE SET expansion = excluded.expansion
                """
            ), {"chat_id": chat_id, "interest": interest, "expansion": expansion})

    def set_category_keyness(self, category: str, scores: list[tuple[str, float]]) -> None:
        """Replaces `category`'s entire row set atomically -- delete then
        insert, not an upsert, since this is a full recompute every
        ingest cycle, not an incrementally-accumulated cache."""
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM category_keyness WHERE category = :category"), {"category": category})
            for term, score in scores:
                conn.execute(text(
                    "INSERT INTO category_keyness (category, term, score) VALUES (:category, :term, :score)"
                ), {"category": category, "term": term, "score": score})

    def get_category_keyness(self, category: str) -> list[tuple]:
        with self._engine.begin() as conn:
            return conn.execute(
                text("SELECT term, score FROM category_keyness WHERE category = :category"),
                {"category": category},
            ).fetchall()
