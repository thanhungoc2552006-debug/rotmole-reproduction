from datasets import load_dataset, concatenate_datasets


def load_commonsenseqa(split="train"):
    ds = load_dataset("tau/commonsense_qa", split=split)

    def convert(ex):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]

        return {
            "task": "commonsenseqa",
            "question": ex["question"],
            "choices": texts,
            "answer_idx": labels.index(ex["answerKey"]),
        }

    return ds.map(
        convert,
        remove_columns=ds.column_names,
    )


def load_openbookqa(split="train"):
    ds = load_dataset(
        "allenai/openbookqa",
        "main",
        split=split,
    )

    def convert(ex):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]

        return {
            "task": "openbookqa",
            "question": ex["question_stem"],
            "choices": texts,
            "answer_idx": labels.index(ex["answerKey"]),
        }

    return ds.map(
        convert,
        remove_columns=ds.column_names,
    )


def load_socialiqa(split="train"):
    revision = "f96e243fd5228e9431b13b9f3bba849a8ec1d013"

    data_files = {
        "train": (
            f"https://huggingface.co/datasets/allenai/social_i_qa/"
            f"resolve/{revision}/data/train-00000-of-00001.parquet"
        ),
        "validation": (
            f"https://huggingface.co/datasets/allenai/social_i_qa/"
            f"resolve/{revision}/data/validation-00000-of-00001.parquet"
        ),
    }

    ds = load_dataset(
        "parquet",
        data_files={split: data_files[split]},
        split=split,
    )

    def convert(ex):
        question = (
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}"
        )

        choices = [
            ex["answerA"],
            ex["answerB"],
            ex["answerC"],
        ]

        # SocialIQA labels are strings: "1", "2", "3"
        answer_idx = int(ex["label"]) - 1

        return {
            "task": "socialiqa",
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
        }

    return ds.map(
        convert,
        remove_columns=ds.column_names,
    )


def load_mixed_qa(split="train"):
    datasets = [
        load_commonsenseqa(split),
        load_openbookqa(split),
        load_socialiqa(split),
    ]

    return concatenate_datasets(datasets)


def format_mcq(example):
    """
    Unified text representation.
    We keep the answer as the option letter.
    """

    letters = ["A", "B", "C", "D", "E"]

    options = "\n".join(
        f"{letters[i]}. {choice}"
        for i, choice in enumerate(example["choices"])
    )

    prompt = (
        f"{example['question']}\n"
        f"{options}\n"
        f"Answer:"
    )

    answer = letters[example["answer_idx"]]

    return {
        "prompt": prompt,
        "answer": answer,
    }
    
class BalancedTaskBatchSampler(Sampler):
    """
    Each micro-batch contains exactly the same number
    of samples from each task.

    Example:
        per_task_batch = 3

    => 3 CSQA + 3 OBQA + 3 SIQA = batch size 9
    """

    def __init__(
        self,
        task_lengths,
        per_task_batch=3,
        num_batches=1000,
        seed=42,
    ):
        self.task_lengths = list(task_lengths)
        self.per_task_batch = per_task_batch
        self.num_batches = num_batches
        self.seed = seed

        # Global index offsets after concatenate_datasets()
        self.offsets = []

        offset = 0
        for n in self.task_lengths:
            self.offsets.append(offset)
            offset += n

    def __len__(self):
        return self.num_batches

    def __iter__(self):

        rng = np.random.default_rng(self.seed)

        permutations = [
            rng.permutation(n)
            for n in self.task_lengths
        ]

        positions = [
            0 for _ in self.task_lengths
        ]

        for _ in range(self.num_batches):

            batch_indices = []

            for task_id, task_len in enumerate(self.task_lengths):

                selected = []
                remaining = self.per_task_batch

                while remaining > 0:

                    # Dataset exhausted -> reshuffle
                    if positions[task_id] >= task_len:
                        permutations[task_id] = rng.permutation(task_len)
                        positions[task_id] = 0

                    available = (
                        task_len
                        - positions[task_id]
                    )

                    take = min(
                        remaining,
                        available,
                    )

                    local_idx = permutations[task_id][
                        positions[task_id]:
                        positions[task_id] + take
                    ]

                    global_idx = (
                        local_idx
                        + self.offsets[task_id]
                    )

                    selected.extend(
                        global_idx.tolist()
                    )

                    positions[task_id] += take
                    remaining -= take

                batch_indices.extend(selected)

            # Mix tasks inside micro-batch
            rng.shuffle(batch_indices)

            yield batch_indices


class MixedQACollator:
    """
    Converts normalized Mixed-QA examples into
    causal-LM training tensors.

    Loss is computed ONLY on the answer tokens.
    """

    def __init__(
        self,
        tokenizer,
        max_length=512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, examples):

        all_input_ids = []
        all_labels = []
        tasks = []

        for ex in examples:

            formatted = format_mcq(ex)

            prompt = formatted["prompt"]

            # Add a space before A/B/C/D...
            answer = " " + formatted["answer"]

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            answer_ids = self.tokenizer(
                answer,
                add_special_tokens=False,
            )["input_ids"]

            # Train model to emit EOS after answer
            answer_ids = (
                answer_ids
                + [self.tokenizer.eos_token_id]
            )

            # Keep room for answer
            max_prompt_len = (
                self.max_length
                - len(answer_ids)
            )

            prompt_ids = prompt_ids[
                :max_prompt_len
            ]

            input_ids = (
                prompt_ids
                + answer_ids
            )

            # Ignore prompt in loss
            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

            # VERY IMPORTANT for q_i analysis later
            tasks.append(ex["task"])

        max_len = max(
            len(x)
            for x in all_input_ids
        )

        padded_input_ids = []
        padded_attention = []
        padded_labels = []

        for input_ids, labels in zip(
            all_input_ids,
            all_labels,
        ):

            pad_len = (
                max_len
                - len(input_ids)
            )

            padded_input_ids.append(
                input_ids
                + [self.tokenizer.pad_token_id]
                * pad_len
            )

            padded_attention.append(
                [1] * len(input_ids)
                + [0] * pad_len
            )

            padded_labels.append(
                labels
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                padded_input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                padded_attention,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                padded_labels,
                dtype=torch.long,
            ),

            # Keep sample-level task identity.
            # Do NOT send this directly into model.forward()
            "tasks": tasks,
        }
