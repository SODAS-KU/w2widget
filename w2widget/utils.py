from typing import List, Callable
from functools import partial


def tokenize_with_ws(text: str, tokenizer: Callable[[str], List[str]]) -> List[str]:
    "Tokenize a string with whitespaces with the specified tokenizer"
    if isinstance(text, str):
        return [
            x for y in [tokenizer(token) + [" "] for token in text.split()] for x in y
        ]


def tokenizer_with_ws(
    tokenizer: Callable[[str], List[str]]
) -> Callable[[str], List[str]]:
    "Returns a function which tokenizes a string with whitespaces based on the specified tokenizer"
    return partial(tokenize_with_ws, tokenizer=tokenizer)

