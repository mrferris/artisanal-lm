import numpy as np
import numpy.random as random
import torch
from jaxtyping import Int
from numpy.typing import NDArray


def load_batch(
    tokens: NDArray,
    batch_size: int,
    context_length: int,
    device="cpu",
) -> tuple[Int[torch.Tensor, "batch_size context_length"]]:
    """
    Load data from sequential token integers into to tensors ready for model input.
    Args:
        tokens: a numpy array of integer tokens
        batch_size: the number of example to split the data into
        context_length: the length of each example
        device_string: device on which to place the resulting tensors
    Returns:
        A tuple of tensors, each of size batch_size x context_length
        The first containing the token sequence examples
        The second containing the correct next token prediction
    """

    # Generate random sample indices
    max_index = len(tokens) - context_length - 1
    random_indices = random.randint(0, max_index + 1, size=batch_size)

    # Gather B contiguous windows on CPU, then issue one transfer to MPS.  The
    # previous loop performed 2*B NumPy copies, device transfers, and device
    # slice assignments per training step.
    offsets = np.arange(context_length + 1)
    windows = np.asarray(tokens[random_indices[:, None] + offsets[None, :]])
    batch = torch.from_numpy(windows).to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


class ConversationBatchLoader:
    def __init__(self, file_path: str, batch_size: int, context_length: int, device: torch.device):
        self.tokens = np.load(file_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        # Define special tokens
        self.END_TOKEN = 0
        self.ME_TOKEN = 1
        self.THEM_TOKEN = 2
        self.CONVERSATION_START_TOKEN = 3

        # Precompute conversation chunks for speed
        self.chunks = self._compute_chunks()

    def _compute_chunks(self) -> list[tuple[int, int]]:
        """
        Split the dataset into chunks that:
        1. Start at <|ConversationStart|> when beginning a new conversation
        2. Continue from previous chunk when a conversation exceeds context_length
        3. Never split individual messages (between special tokens)
        4. Are randomly shuffled so there's no temporal bias during training

        Returns:
            List of (start_index, length) tuples for each chunk
        """
        chunks = []
        i = 0

        while i < len(self.tokens):
            # Skip endoftext tokens
            if self.tokens[i] == self.END_TOKEN:
                i += 1
                continue

            # Check if this is a conversation start or a continuation point
            if self.tokens[i] == self.CONVERSATION_START_TOKEN:
                # Start of a new conversation
                conversation_start = i
                current_pos = i

                # Collect messages until we hit another CS or endoftext
                messages = []  # List of (start, end) for each message

                while current_pos < len(self.tokens):
                    if self.tokens[current_pos] == self.END_TOKEN:
                        break
                    if current_pos != conversation_start and self.tokens[current_pos] == self.CONVERSATION_START_TOKEN:
                        # Hit the next conversation
                        break

                    if self.tokens[current_pos] in (self.ME_TOKEN, self.THEM_TOKEN, self.CONVERSATION_START_TOKEN):
                        msg_start = current_pos
                        current_pos += 1
                        while current_pos < len(self.tokens) and self.tokens[current_pos] not in (
                            self.ME_TOKEN,
                            self.THEM_TOKEN,
                            self.CONVERSATION_START_TOKEN,
                            self.END_TOKEN,
                        ):
                            current_pos += 1
                        messages.append((msg_start, current_pos))
                    else:
                        current_pos += 1

                # Now split messages into chunks that fit in context_length
                chunk_start = conversation_start
                chunk_token_count = 0

                for msg_start, msg_end in messages:
                    msg_length = msg_end - msg_start

                    if msg_length > self.context_length:
                        # Flush, then slice the over-long message to fit the context window.
                        if chunk_token_count > 0:
                            chunks.append((chunk_start, chunk_token_count))
                            chunk_token_count = 0
                        pos = msg_start
                        while pos < msg_end:
                            piece = min(self.context_length, msg_end - pos)
                            chunks.append((pos, piece))
                            pos += piece
                        chunk_start = msg_end
                    elif chunk_token_count + msg_length > self.context_length:
                        # Save current chunk and start a new one
                        if chunk_token_count > 0:
                            chunks.append((chunk_start, chunk_token_count))
                        chunk_start = msg_start
                        chunk_token_count = msg_length
                    else:
                        chunk_token_count += msg_length

                # Save the final chunk of this conversation
                if chunk_token_count > 0:
                    chunks.append((chunk_start, chunk_token_count))

                i = current_pos
            else:
                i += 1

        return chunks

    def load_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load a batch by randomly sampling from pre-computed conversation chunks.

        Returns:
            padded_seqs: (batch, context_length) tensor of inputs
            padded_labels: (batch, context_length) tensor of labels
        """
        if len(self.chunks) == 0:
            raise ValueError("No conversation chunks found in dataset")

        # Randomly sample chunks for this batch
        chosen_chunk_indices = np.random.choice(len(self.chunks), self.batch_size, replace=True)

        batch_sequences = []
        batch_labels = []

        for chunk_idx in chosen_chunk_indices:
            start_idx, length = self.chunks[chunk_idx]

            # Extract the chunk from the token array
            chunk_tokens = self.tokens[start_idx : start_idx + length]

            # Convert to tensor
            seq = torch.from_numpy(chunk_tokens.copy()).to(self.device, dtype=torch.long)

            # Labels are shifted by 1 (predict next token)
            label = torch.zeros_like(seq)
            label[:-1] = seq[1:]
            label[-1] = self.END_TOKEN

            batch_sequences.append(seq)
            batch_labels.append(label)

        # Pad sequences to the same length (the max length in this batch)
        lengths = [len(seq) for seq in batch_sequences]
        max_len = max(lengths)

        padded_seqs = torch.full((self.batch_size, max_len), self.END_TOKEN, dtype=torch.long, device=self.device)
        padded_labels = torch.full((self.batch_size, max_len), self.END_TOKEN, dtype=torch.long, device=self.device)

        for i, (seq, label) in enumerate(zip(batch_sequences, batch_labels)):
            padded_seqs[i, : lengths[i]] = seq
            padded_labels[i, : lengths[i]] = label

        return padded_seqs, padded_labels
