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
    def __init__(
        self,
        file_path: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
        seed: int | None = None,
        token_loss_weights: dict[int, int] | None = None,
    ):
        self.tokens = np.load(file_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.token_loss_weights = token_loss_weights or {}

        # Define special tokens
        self.END_TOKEN = 0
        self.ME_TOKEN = 1
        self.THEM_TOKEN = 2
        self.CONVERSATION_START_TOKEN = 3

        # Precompute conversation chunks for speed
        self.chunks = self._compute_chunks()
        self.chunk_lengths = np.asarray(
            [length for _, length in self.chunks], dtype=np.int32
        )
        self.packed_chunks = self._pack_chunks()

    def _pack_chunks(self) -> NDArray:
        """Materialize input, shifted target, and selective loss weights once."""
        packed = np.full(
            (3, len(self.chunks), self.context_length),
            self.END_TOKEN,
            dtype=self.tokens.dtype,
        )
        for row, (start_idx, length) in enumerate(self.chunks):
            chunk = self.tokens[start_idx : start_idx + length]
            packed[0, row, :length] = chunk
            # An over-long message can be sliced into continuation chunks that
            # begin with content rather than a role marker. Recover the role
            # immediately preceding that slice so its content is not silently
            # omitted from the Me-only objective.
            active_speaker = None
            if int(chunk[0]) not in (
                self.END_TOKEN,
                self.ME_TOKEN,
                self.THEM_TOKEN,
                self.CONVERSATION_START_TOKEN,
            ):
                previous = start_idx - 1
                while previous >= 0:
                    token = int(self.tokens[previous])
                    if token in (self.ME_TOKEN, self.THEM_TOKEN):
                        active_speaker = token
                        break
                    if token in (
                        self.END_TOKEN,
                        self.CONVERSATION_START_TOKEN,
                    ):
                        break
                    previous -= 1
            for position in range(length):
                input_token = int(chunk[position])
                if input_token == self.ME_TOKEN:
                    active_speaker = self.ME_TOKEN
                elif input_token == self.THEM_TOKEN:
                    active_speaker = self.THEM_TOKEN
                elif input_token in (
                    self.END_TOKEN,
                    self.CONVERSATION_START_TOKEN,
                ):
                    active_speaker = None

                # Use the real following token even at a chunk boundary. The
                # former synthetic EOT target made every capacity split look
                # like a conversation ending once structural loss was enabled.
                next_position = start_idx + position + 1
                target_token = (
                    int(self.tokens[next_position])
                    if next_position < len(self.tokens)
                    else self.END_TOKEN
                )
                packed[1, row, position] = target_token

                target_is_structure = target_token in (
                    self.ME_TOKEN,
                    self.THEM_TOKEN,
                    self.END_TOKEN,
                )
                target_is_me_content = (
                    active_speaker == self.ME_TOKEN
                    and not target_is_structure
                    and target_token != self.CONVERSATION_START_TOKEN
                )
                if target_is_structure or target_is_me_content:
                    packed[2, row, position] = self.token_loss_weights.get(
                        target_token, 1
                    )
        return packed

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

    def load_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load a batch by randomly sampling from pre-computed conversation chunks.

        Returns:
            padded_seqs: (batch, context_length) tensor of inputs
            padded_labels: (batch, context_length) tensor of labels
        """
        if len(self.chunks) == 0:
            raise ValueError("No conversation chunks found in dataset")

        # Randomly sample chunks for this batch
        chosen_chunk_indices = self.rng.choice(
            len(self.chunks), self.batch_size, replace=True
        )

        max_len = int(self.chunk_lengths[chosen_chunk_indices].max())
        packed = np.ascontiguousarray(
            self.packed_chunks[:, chosen_chunk_indices, :max_len]
        )
        batch = torch.from_numpy(packed).to(device=self.device, dtype=torch.long)
        return batch[0], batch[1], batch[2]


class ResponseBatchLoader:
    """Sample one Me response burst with its preceding conversation context.

    A response burst is one or more consecutive <|Me|> messages. Each window
    contains as many complete preceding messages as fit, always including the
    immediately preceding Them message when the fixed context length permits.
    Loss is zero on that context and non-zero only on the response burst plus
    its following Them/EOT boundary.
    """

    END_TOKEN = 0
    ME_TOKEN = 1
    THEM_TOKEN = 2
    CONVERSATION_START_TOKEN = 3

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
        seed: int | None = None,
        token_loss_weights: dict[int, int] | None = None,
        response_prefix_tokens: int = 0,
        response_prefix_weight: int = 1,
    ):
        self.tokens = np.load(file_path, mmap_mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.token_loss_weights = token_loss_weights or {}
        if response_prefix_tokens < 0:
            raise ValueError("response_prefix_tokens must be non-negative")
        if response_prefix_weight < 1:
            raise ValueError("response_prefix_weight must be at least 1")
        self.response_prefix_tokens = response_prefix_tokens
        self.response_prefix_weight = response_prefix_weight
        # (window_start, input_length, target_burst_start, target_burst_end)
        self.windows = self._compute_windows()
        if not self.windows:
            raise ValueError("No Me response bursts found in dataset")
        self.window_lengths = np.asarray(
            [length for _, length, _, _ in self.windows], dtype=np.int32
        )
        self.packed_windows = self._pack_windows()

    def _conversation_messages(
        self, conversation_start: int, conversation_end: int
    ) -> list[tuple[int, int, int]]:
        """Return (start, end, role) messages, with role marker at start."""
        messages = []
        position = conversation_start + 1
        while position < conversation_end:
            role = int(self.tokens[position])
            if role not in (self.ME_TOKEN, self.THEM_TOKEN):
                position += 1
                continue
            end = position + 1
            while end < conversation_end and int(self.tokens[end]) not in (
                self.ME_TOKEN,
                self.THEM_TOKEN,
                self.CONVERSATION_START_TOKEN,
                self.END_TOKEN,
            ):
                end += 1
            messages.append((position, end, role))
            position = end
        return messages

    def _compute_windows(self) -> list[tuple[int, int, int, int]]:
        windows = []
        position = 0
        token_count = len(self.tokens)
        while position < token_count:
            if int(self.tokens[position]) != self.CONVERSATION_START_TOKEN:
                position += 1
                continue
            conversation_start = position
            conversation_end = position + 1
            while conversation_end < token_count and int(
                self.tokens[conversation_end]
            ) not in (self.END_TOKEN, self.CONVERSATION_START_TOKEN):
                conversation_end += 1

            messages = self._conversation_messages(
                conversation_start, conversation_end
            )
            message_index = 0
            while message_index < len(messages):
                if messages[message_index][2] != self.ME_TOKEN:
                    message_index += 1
                    continue

                burst_first = message_index
                burst_last = message_index
                while (
                    burst_last + 1 < len(messages)
                    and messages[burst_last + 1][2] == self.ME_TOKEN
                ):
                    burst_last += 1
                burst_start = messages[burst_first][0]
                burst_end = messages[burst_last][1]

                # Normal responses fit in one window. Choose the earliest
                # complete-message boundary that retains the response.
                if burst_end - burst_start <= self.context_length:
                    window_start = burst_start
                    if burst_end - conversation_start <= self.context_length:
                        window_start = conversation_start
                    else:
                        viable_messages = [
                            message
                            for message in messages[:burst_first]
                            if burst_end - message[0] <= self.context_length
                        ]
                        # A Them boundary makes the response-learning task
                        # explicit. Prefer the earliest fitting one; only fall
                        # back to a Me boundary for opener/exceptional cases.
                        viable_them = [
                            message
                            for message in viable_messages
                            if message[2] == self.THEM_TOKEN
                        ]
                        if viable_them:
                            window_start = viable_them[0][0]
                        elif viable_messages:
                            window_start = viable_messages[0][0]
                    windows.append(
                        (
                            window_start,
                            burst_end - window_start,
                            burst_start,
                            burst_end,
                        )
                    )
                else:
                    # An exceptionally long response cannot retain its original
                    # prompt throughout. Split it causally; the first segment
                    # retains as much preceding context as possible and later
                    # segments condition on the response prefix.
                    segment_end = burst_start + self.context_length
                    first = True
                    while segment_end < burst_end:
                        segment_start = max(
                            conversation_start,
                            segment_end - self.context_length,
                        )
                        windows.append(
                            (
                                segment_start,
                                segment_end - segment_start,
                                burst_start if first else segment_start,
                                segment_end,
                            )
                        )
                        first = False
                        segment_end += self.context_length
                    segment_start = max(
                        conversation_start, burst_end - self.context_length
                    )
                    windows.append(
                        (
                            segment_start,
                            burst_end - segment_start,
                            max(burst_start, segment_start),
                            burst_end,
                        )
                    )

                message_index = burst_last + 1
            position = max(conversation_end, position + 1)
        return windows

    def _pack_windows(self) -> NDArray:
        packed = np.full(
            (3, len(self.windows), self.context_length),
            self.END_TOKEN,
            dtype=self.tokens.dtype,
        )
        for row, (start, length, burst_start, burst_end) in enumerate(
            self.windows
        ):
            packed[0, row, :length] = self.tokens[start : start + length]
            prefix_tokens_remaining = 0
            for local_position in range(length):
                target_position = start + local_position + 1
                target = (
                    int(self.tokens[target_position])
                    if target_position < len(self.tokens)
                    else self.END_TOKEN
                )
                packed[1, row, local_position] = target
                # Supervise the response itself and exactly one real boundary
                # after it. Context—including earlier Me messages—is mask zero.
                selected = (
                    burst_start <= target_position < burst_end
                    or target_position == burst_end
                )
                if selected:
                    # Begin a fresh positional emphasis region after every Me
                    # marker, including consecutive Me bubbles in one burst.
                    # Role/boundary tokens themselves retain their base weight.
                    prefix_multiplier = 1
                    if target == self.ME_TOKEN:
                        prefix_tokens_remaining = self.response_prefix_tokens
                    elif target not in (
                        self.THEM_TOKEN,
                        self.CONVERSATION_START_TOKEN,
                        self.END_TOKEN,
                    ):
                        if prefix_tokens_remaining > 0:
                            prefix_multiplier = self.response_prefix_weight
                            prefix_tokens_remaining -= 1
                    packed[2, row, local_position] = (
                        self.token_loss_weights.get(target, 1)
                        * prefix_multiplier
                    )
        return packed

    def load_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = self.rng.choice(
            len(self.windows), self.batch_size, replace=True
        )
        max_len = int(self.window_lengths[chosen].max())
        packed = np.ascontiguousarray(
            self.packed_windows[:, chosen, :max_len]
        )
        batch = torch.from_numpy(packed).to(
            device=self.device, dtype=torch.long
        )
        return batch[0], batch[1], batch[2]
