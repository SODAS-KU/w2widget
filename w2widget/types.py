from typing import TypedDict


class Topic(TypedDict):
    topic_words: list[str]
    search_words: list[str]
    negative_words: list[str]
    skip_words: list[str]
