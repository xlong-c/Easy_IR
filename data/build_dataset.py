def build_dataset(dataset_name, path, mode: str, to_bad_fn, transform=None):
    if dataset_name== 'mri':
        from data.dataset_mri import Brain_data
        dataset = Brain_data(path, mode, to_bad_fn, transform)
    elif dataset_name == 'diff_mri':
        from data.dataset_mri_diff import Brain_data
        dataset = Brain_data(path, mode, to_bad_fn, transform)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_name}")
    return dataset
