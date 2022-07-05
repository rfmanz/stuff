def drop_duplicated_columns_after_merge(df):
    """
    drop duplicated columns after merge

    if two dfs have identical column names before merge, columns are appended with
    _x and _y for differentiation. Use this helper to check whether the two is the same
    and drop one if so.
    """

    candidate_pairs, mapper = [], {}
    for (c1, c2) in zip(cols[:-1], cols[1:]):
        if c1.endswith('_x') and c2.endswith('_y') and c1[:-2] == c2[:-2]:
            candidate_pairs.append((c1, c2))
            # make sure the two columns are the same before exclusion
            if df[c1].equals(df[c2]):
                # drop one
                mapper[c1] = c1[:-2]
                df.drop(labels=[c2], axis=1, inplace=True)
            else:
                print(f'{c1} and {c2} are not equivalent, keeping both columns')
    df.rename(mapper, inplace=True)
    return df
