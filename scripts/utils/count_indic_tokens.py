import argparse

from transformers import AutoTokenizer

# Define language-specific Unicode ranges
language_unicode_ranges = {
    'European': ('\u00', '\u007F'),
    'Chinese (Basic)': ('\u4E00', '\u9FFF'),
    'Tamil': ('\u0B80', '\u0BFF'),
    'Hindi': ('\u09', '\u097F'),
    'Telugu': ('\u0C00', '\u0C7F'),
    'Malayalam': ('\u0D00', '\u0D7F'),
    'Kannada': ('\u0C80', '\u0CFF'),
    'Marathi': ('\u09', '\u097F'),  # Marathi shares the range with Hindi
    'Bengali': ('\u0980', '\u09FF'),
}

def is_language(token, ranges):
    """
    Check if a token falls within a specific Unicode range.

    Args:
        token (str): Token to check.
        ranges (tuple): Tuple containing start and end Unicode characters.

    Returns:
        bool: True if the token falls within the specified range, False otherwise.
    """
    return any(ranges[0] <= char <= ranges[1] for char in token)

def count_language_tokens(tokenizer, ranges):
    """
    Count tokens that fall within a specific Unicode range.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer to query.
        ranges (tuple): Tuple containing start and end Unicode characters.

    Returns:
        int: Number of tokens that fall within the specified range.
    """
    return sum(is_language(token, ranges) for token in tokenizer.get_vocab().keys())

def main(model_name):
    """
    Main function to count language-specific tokens and their percentage in a tokenizer's vocabulary.

    Args:
        model_name (str): Name of the model to analyze.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_vocab_size = len(tokenizer.get_vocab())

    print("\n---Note: These calculations are approximate!---\n")
    print(f"Total vocabulary size of '{model_name}': {total_vocab_size}\n")
    print(f"{'Language':<20} | {'Tokens':>10} | {'Percentage':>10}")
    print("-" * 50)

    for language, ranges in language_unicode_ranges.items():
        count = count_language_tokens(tokenizer, ranges)
        percentage = (count / total_vocab_size) * 100
        print(f"{language:<20} | {count:>10} | {percentage:>9.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count language-specific tokens and their percentage in a tokenizer's vocabulary.")
    parser.add_argument("model_name", type=str, help="Name of the model to analyze")

    args = parser.parse_args()

    main(args.model_name)
