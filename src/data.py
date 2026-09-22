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
