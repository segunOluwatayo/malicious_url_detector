from sklearn.model_selection import GroupShuffleSplit

def group_split(urls, labels, *, test_size=0.3, val_size=0.15, random_state=42):
    """
    Domain-level train/val/test indices (no leakage).
    Returns (train_idx, val_idx, test_idx)
    """
    import tldextract, numpy as np

    domains = np.array(
        [tldextract.extract(u).registered_domain for u in urls], dtype=object
    )

    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_idx, tmp_idx = next(gss.split(urls, labels, groups=domains))

    tmp_domains = domains[tmp_idx]
    tmp_labels = labels[tmp_idx]
    gss2 = GroupShuffleSplit(
        test_size=val_size / (1 - test_size), random_state=random_state
    )
    val_idx, test_idx = next(gss2.split(tmp_idx, tmp_labels, groups=tmp_domains))

    # map val_idx/test_idx back to original index space
    val_idx = tmp_idx[val_idx]
    test_idx = tmp_idx[test_idx]
    return train_idx, val_idx, test_idx
