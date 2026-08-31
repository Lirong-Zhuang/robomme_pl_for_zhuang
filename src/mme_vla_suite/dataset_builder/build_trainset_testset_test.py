from scripts.build_trainset_testset import split_episode_indices


def test_split_episode_indices_is_deterministic_and_disjoint():
    candidates = {
        "BinFill": list(range(100)),
        "PickXtimes": list(range(100)),
    }

    train_a, test_a = split_episode_indices(
        candidates,
        test_ratio=0.1,
        seed=42,
    )
    train_b, test_b = split_episode_indices(
        candidates,
        test_ratio=0.1,
        seed=42,
    )

    assert (train_a, test_a) == (train_b, test_b)
    for task_name in candidates:
        assert len(train_a[task_name]) == 90
        assert len(test_a[task_name]) == 10
        assert set(train_a[task_name]).isdisjoint(test_a[task_name])
        assert sorted(train_a[task_name] + test_a[task_name]) == candidates[task_name]


def test_task_split_does_not_depend_on_other_selected_tasks():
    binfill_only = {"BinFill": list(range(100))}
    train_only, test_only = split_episode_indices(
        binfill_only,
        test_ratio=0.1,
        seed=7,
    )
    train_many, test_many = split_episode_indices(
        {**binfill_only, "PickXtimes": list(range(100))},
        test_ratio=0.1,
        seed=7,
    )

    assert train_only["BinFill"] == train_many["BinFill"]
    assert test_only["BinFill"] == test_many["BinFill"]


def test_split_ratio_scales_with_dataset_size_and_rounds_up():
    train, test = split_episode_indices(
        {"BinFill": list(range(125))},
        test_ratio=0.1,
        seed=42,
    )

    assert len(train["BinFill"]) == 112
    assert len(test["BinFill"]) == 13
