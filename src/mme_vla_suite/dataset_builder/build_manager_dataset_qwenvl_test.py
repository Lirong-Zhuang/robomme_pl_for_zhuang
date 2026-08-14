from mme_vla_suite.dataset_builder.build_manager_dataset_qwenvl import DatasetBuilder


def test_select_idxs_always_include_short_subgoal_transitions():
    transition_idxs = [0, 8, 20]

    select_idxs, duplicate_idxs = DatasetBuilder._compute_select_and_duplicate_idxs(
        None,
        transition_idxs,
        num_timesteps=21,
        env_id="BinFill",
    )

    assert select_idxs == transition_idxs
    assert duplicate_idxs == {8: 0}


def test_forced_transitions_preserve_interval_sampling_and_duplication():
    transition_idxs = [0, 40, 50, 100]

    select_idxs, duplicate_idxs = DatasetBuilder._compute_select_and_duplicate_idxs(
        None,
        transition_idxs,
        num_timesteps=101,
        env_id="BinFill",
    )

    assert select_idxs == [0, 20, 40, 50, 66, 83, 100]
    assert duplicate_idxs == {40: 1, 50: 0}
