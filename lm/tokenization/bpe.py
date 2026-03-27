import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from collections.abc import Callable
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO

import numpy as np
import regex as re

PRE_TOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PRE_TOKENIZATION_REGEX = re.compile(PRE_TOKENIZATION_REGEX)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
    token_callback: Callable[[int, bytes], None] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Creates a BPE tokenizer vocabulary and its ordered merges.

    If ``progress_callback`` is given, it is called as ``progress_callback(done, total)``
    periodically during the merge loop (and once at completion) so callers can report
    determinate progress.

    If ``token_callback`` is given, it is called as ``token_callback(token_id, token_bytes)``
    for every newly-created merge token, in id order, so callers can stream the vocabulary
    as it is built.
    """
    # Vocabulary Initialization:
    vocab: dict[int, bytes] = {}
    next_id = 0
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # Pre-tokenization
    word_counts = _get_pre_tokenized_data(input_path, special_tokens)

    # Metadata
    word_bytes = {word: [bytes([b]) for b in word.encode("utf-8")] for word in word_counts}
    pair_counts = _get_pair_counts(word_bytes, word_counts)
    pair_to_words = _build_pair_to_words_index(word_bytes)

    # Keep track of the values that were merged
    merges: list[tuple[bytes, bytes]] = []
    max_merges = vocab_size - len(vocab)

    for i in range(max_merges):
        if not pair_counts:
            break
        most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_token = most_common_pair[0] + most_common_pair[1]

        _merge_vocab_bytes_with_index(most_common_pair, word_bytes, word_counts, pair_counts, pair_to_words, new_token)
        merges.append(most_common_pair)

        vocab[next_id] = new_token
        if token_callback is not None:
            token_callback(next_id, new_token)
        next_id += 1

        if progress_callback is not None and (i % 50 == 0 or i == max_merges - 1):
            progress_callback(len(merges), max_merges)

    if progress_callback is not None:
        progress_callback(len(merges), max_merges)

    return (vocab, merges)


def _get_pre_tokenized_data(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[str]:
    """
    Splits a corpus into pretokens (to be further tokenized by BPE).
    """

    num_processes = cpu_count()

    with open(input_path, "rb") as f:
        # Handle empty special_tokens case
        if special_tokens:
            split_token = special_tokens[0].encode("utf-8")
            boundaries = _find_chunk_boundaries(f, num_processes, split_token)
        else:
            # Use uniform chunking when no special tokens
            file_size = os.path.getsize(input_path)
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            boundaries[-1] = file_size

        args = []
        for i in range(len(boundaries) - 1):
            args.append((input_path, special_tokens, boundaries[i], boundaries[i + 1]))

        with Pool(num_processes) as pool:
            results = pool.map(_process_chunk, args)

        aggregated_counter = Counter()

        for counter in results:
            aggregated_counter.update(counter)

        return aggregated_counter


def _process_chunk(args: tuple[str, list[str], int, int]) -> Counter[str]:
    """
    Pretokenize a single chunk of text and return the counted words.
    """
    input_path, special_tokens, begin_index, end_index = args

    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    escaped_special_tokens = "|".join(escaped_special_tokens)
    compiled_escaped_special_tokens = re.compile(f"({escaped_special_tokens})")

    with open(input_path, "br") as f:
        f.seek(begin_index)
        chunk_text = f.read(end_index - begin_index).decode("utf-8", errors="ignore")

        counted_words: Counter[tuple[bytes]] = Counter()
        split_text = compiled_escaped_special_tokens.split(chunk_text)
        for split in split_text:
            if split not in special_tokens and split.strip():
                for match in COMPILED_PRE_TOKENIZATION_REGEX.finditer(split):
                    counted_words[match.group()] += 1

        return counted_words


def _find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _get_pair_counts(word_bytes: dict[str, list[bytes]], word_freqs: dict[str, int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count instances of each consecutive byte pair across the dataset.

    Args:
        word_splits: Mapping from word to its byte sequences
        word_freqs: Count of each word in the corpus

    Returns:
        Dictionary mapping byte pairs to their frequencies
    """
    pairs = defaultdict(int)

    for word, freq in word_freqs.items():
        symbols = word_bytes[word]
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return dict(pairs)


def _build_pair_to_words_index(word_bytes: dict[str, list[bytes]]) -> dict[tuple[bytes, bytes], set[str]]:
    """
    Build an index mapping pairs to the words that contain them.

    Args:
        word_splits: Mapping from words to their byte sequences

    Returns:
        Dictionary mapping each pair to a set of words that contain it
    """
    pair_to_words = defaultdict(set)

    for word, symbols in word_bytes.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_to_words[pair].add(word)

    return dict(pair_to_words)


def _merge_vocab_bytes_with_index(
    pair: tuple[bytes, bytes],
    word_splits: dict[str, list[bytes]],
    word_freqs: dict[str, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[str]],
    new_token: bytes,
) -> None:
    """
    Merge all occurences of a byte pair using reverse index for maximum efficiency.

    Args:
        pair: The byte pair to merge
        word_splits: Current word splits
        word_freqs: Frequency of each word
        pair_counts: Current pair counts
        pair_to_words: Reverse index from pairs to words
        new_token: The merged token to replace pair with
    """
    affected_words = list(pair_to_words.get(pair, set()))

    if pair in pair_counts:
        del pair_counts[pair]
    if pair in pair_to_words:
        del pair_to_words[pair]

    pair0, pair1 = pair

    for word in affected_words:
        old_symbols = word_splits[word]
        freq = word_freqs[word]

        for i in range(len(old_symbols) - 1):
            old_pair = (old_symbols[i], old_symbols[i + 1])
            if old_pair in pair_counts:
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

            if old_pair in pair_to_words:
                pair_to_words[old_pair].discard(word)
                if not pair_to_words[old_pair]:
                    del pair_to_words[old_pair]

        new_symbols = []
        i = 0
        while i < len(old_symbols):
            if i < len(old_symbols) - 1 and old_symbols[i] == pair0 and old_symbols[i + 1] == pair1:
                new_symbols.append(new_token)
                i += 2
            else:
                new_symbols.append(old_symbols[i])
                i += 1

        word_splits[word] = new_symbols

        for i in range(len(new_symbols) - 1):
            new_pair = (new_symbols[i], new_symbols[i + 1])
            if new_pair in pair_counts:
                pair_counts[new_pair] += freq
            else:
                pair_counts[new_pair] = freq

            if new_pair not in pair_to_words:
                pair_to_words[new_pair] = set()
            pair_to_words[new_pair].add(word)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []

        self.reverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Ensure special tokens are in vocab
        next_token_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.reverse_vocab:
                self.vocab[next_token_id] = special_bytes
                self.reverse_vocab[special_bytes] = next_token_id
                next_token_id += 1

        # Create merge ranks for efficient BPE encoding
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(self.merges)}

    def save(self, vocab_path: str, merges_path: str):
        serializable_vocab = {str(k): v.hex() for k, v in self.vocab.items()}

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=2)

        with open(merges_path, "wb") as f:
            pickle.dump(self.merges, f)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        if vocab_filepath.endswith(".json"):
            with open(vocab_filepath, encoding="utf-8") as f:
                vocab_data = json.load(f)
                vocab = {}
                for k, v in vocab_data.items():
                    token_id = int(k)
                    if isinstance(v, str):
                        try:
                            vocab[token_id] = bytes.fromhex(v)
                        except ValueError:
                            vocab[token_id] = v.encode("utf-8")
                    elif isinstance(v, list):
                        vocab[token_id] = bytes(v)
                    else:
                        vocab[token_id] = v
        else:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)

        if merges_filepath.endswith(".json"):
            with open(merges_filepath, encoding="utf-8") as f:
                merges_data = json.load(f)
                merges = []
                for merge in merges_data:
                    if isinstance(merge[0], str):
                        merge_tuple = (merge[0].encode("utf-8"), merge[1].encode("utf-8"))
                    elif isinstance(merge[0], list):
                        merge_tuple = (bytes(merge[0]), bytes(merge[1]))
                    else:
                        merge_tuple = merge
                    merges.append(merge_tuple)
        else:
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        splits = self._split(text)

        encoded_text: list[int] = []
        for split in splits:
            if len(split) == 0:
                continue
            if self.special_tokens and split in self.special_tokens:
                encoded_text.append(self.reverse_vocab[split.encode("utf-8")])
            else:
                words = COMPILED_PRE_TOKENIZATION_REGEX.finditer(split)
                for match in words:
                    bpe_encoded = self._encode_text_bytes(tuple(bytes([b]) for b in match.group().encode("utf-8")))
                    encoded_text.extend(bpe_encoded)
        return encoded_text

    def _split(self, text) -> list[str]:
        # Split on special tokens
        splits = [text]
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_token_regex = re.compile(f"({'|'.join(re.escape(special_token) for special_token in sorted_special_tokens)})")
            splits = special_token_regex.split(text)

        return splits

    def _encode_text_bytes(self, text_bytes: tuple[bytes]) -> list[int]:
        """
        Apply BPE merges to a byte sequence using greedy-by-rank algorithm.

        Args:
            text_bytes: Tuple of individual byte tokens to encode

        Returns:
            List of token IDs after applying BPE merges
        """
        if len(text_bytes) <= 1:
            # Single byte or empty - just convert to token IDs
            token_ids = []
            for byte_token in text_bytes:
                if byte_token in self.reverse_vocab:
                    token_ids.append(self.reverse_vocab[byte_token])
                else:
                    # Fallback: split into individual bytes
                    for byte_val in byte_token:
                        single_byte = bytes([byte_val])
                        if single_byte in self.reverse_vocab:
                            token_ids.append(self.reverse_vocab[single_byte])
            return token_ids

        # Convert to list for mutation
        word = list(text_bytes)

        # Repeatedly find and apply the earliest-rank merge until no more merges possible
        while True:
            # Find all valid pairs in current word state
            pairs = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_ranks:
                    pairs.append((self.merge_ranks[pair], i, pair))

            if not pairs:
                break

            # Sort by rank (earliest merge first), then by position for stability
            pairs.sort()
            _, _, (first, second) = pairs[0]

            # Apply the merge with lowest rank
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

        # Convert final tokens to IDs
        token_ids = []
        for token in word:
            if token in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token])
            else:
                # Fallback: split unknown token into individual bytes
                for byte_val in token:
                    single_byte = bytes([byte_val])
                    if single_byte in self.reverse_vocab:
                        token_ids.append(self.reverse_vocab[single_byte])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.

        This method processes the input line by line to minimize memory usage,
        making it suitable for large files that don't fit in memory.

        Args:
            iterable: An iterable of strings (e.g., file handle)

        Yields:
            Token IDs one at a time
        """
        buffer = ""

        for line in iterable:
            buffer += line

            while buffer:
                if len(buffer) > 8192:  # Process in 8KB chunks
                    # Find last newline in first 8KB
                    chunk_end = buffer.rfind("\n", 0, 8192)
                    if chunk_end == -1:
                        # No newline found, take a smaller chunk at word boundary
                        chunk_end = buffer.rfind(" ", 0, 4096)
                        if chunk_end == -1:
                            chunk_end = 4096  # Force split if no word bounary

                    chunk = buffer[: chunk_end + 1]
                    buffer = buffer[chunk_end + 1 :]

                    token_ids = self.encode(chunk)
                    yield from token_ids
                else:
                    break

        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not ids:
            return ""

        decoded_bytes: list[bytes] = []
        for token_id in ids:
            if token_id in self.vocab:
                decoded_bytes.append(self.vocab[token_id])
            # Skip invalid token IDs silently

        return b"".join(decoded_bytes).decode("utf-8", errors="replace")

    def encode_dataset_to_numpy(
        self,
        file_path: str | Path,
        output_path: str | Path,
        max_vocab_size: int = 65536,  # uint16 max value
    ) -> None:
        """
        Encode a large dataset to numpy array with memory-efficient streaming.

        Args:
            tokenizer: The tokenizer to use
            file_path: Path to input text file
            output_path: Path to save encoded numpy array
            chunk_size: Size of chunks to process at once (bytes)
            max_vocab_size: Maximum vocabulary size (for uint16 validation)
        """
        print(f"Encoding dataset {file_path} to {output_path}...")

        all_token_ids: list[int] = []

        with open(file_path, encoding="utf-8") as f:
            token_count = 0
            for token_id in self.encode_iterable(f):
                all_token_ids.append(token_id)
                token_count += 1

                if token_count % 1_000_000 == 0:
                    print(f"  Processed {token_count:,} tokens...")

        print(f"Total tokens: {len(all_token_ids):,}")

        max_token_id = max(all_token_ids)
        if max_token_id >= max_vocab_size:
            raise ValueError(f"Token ID {max_token_id} exceeds uint16 range [0, {max_vocab_size - 1}]")

        token_array = np.array(all_token_ids, dtype=np.uint16)

        np.save(output_path, token_array)

        print(f"Saved {len(token_array):,} tokens to {output_path}")
        print(f"Array shape: {token_array.shape}")
        print(f"Array dtype: {token_array.dtype}")
        print(f"File size: {os.path.getsize(str(output_path)) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-path", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir-path", type=str, required=True)
    parser.add_argument("--vocab-name", type=str, required=True)
    args = parser.parse_args()

    training_data_path = args.training_data_path
    vocab_size = args.vocab_size
    vocab_output_dir_path = args.output_dir_path
    vocab_name = args.vocab_name

    t = Tokenizer.from_files("../vocab/openwebtext_vocab.json", "../output/openwebtext_merges.pkl")
    t.encode_dataset_to_numpy("../data/imessages_sft.txt", "../data/encoded/imessages_sft.npy")
