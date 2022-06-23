#### Money customer level score
---

##### Data Pipeline

`make all` to run the entire pipeline. For individual steps, checkout the `MakeFile` for options.

Development data was created with Dynamic and Static sampling. Since one is on transactional level and the other account level, we separated the process into two src folders:
* `src`
* `src-transactional`

Running the MakeFile would generate two corresponding data folders:
* `data`
* `data-transactional`

Hyperparameters are contained in `config.json` and `config-transactional.json`, which also automatically register the id of the newly queried data and featuers. 

##### Training

Everything is included in `train.py` 

##### Monitoring

Please see `notebooks/monitoring-template` directory.

##### Development Details

See `notebooks`

##### Sampling Methods
---

In production, the model will be evaluated on the entire group users at a given snapshot time, but we cannot simply take this approach when building training data due to the nature of customer level frauds. Explainations can be found [here](https://docs.google.com/presentation/d/1oAsBXqkIpkjSqFkpmajQaD-Km_oySRJfjuejj2S_AsY/edit#slide=id.g7786316f26_1_0).

As the solution, we combine records obtains from both static and dynamic sampling. Static sampling is simple: just choose a time and take available data from all users. Dynamic sampling is carried out by taking multiple (15 in this case) records from each user's account before every snapshot dates. 

* `src/data.py`: contains code for static sampling
    * see function `sample_on_date_range`
    * Recipe: assign sample date to multiple copies of banking account-level data, combine them, and then remove the records that doesn't exist on the sampling dates...good logic but tbh quite memory inefficient. Then merge with other attributes.
    ```python
        for date in dates:
            df["sample_date"] = date
            dfs.append(df.copy())

        sampled_df = pd.concat(dfs, ignore_index=True)
        sampled_df = sampled_df[
            sampled_df["sample_date"] >= sampled_df["date_account_opened"]
        ]
        sampled_df = sampled_df[
            (sampled_df["sample_date"] < sampled_df["date_account_closed"])
            | sampled_df["date_account_closed"].isna()
        ]
    ```
* `src/transactional`: contains data for dynamic sampling. For dynamic sampling we first build data from transactions perspective, which gives us a time-based history for each user. Then be merge account level data toward the transactions. Then as recorded in `src/combine.py` 
    * then for each user, we sample up to 15 records. We do that by first randomly sort the data, then take first 15 transactions made by each user. 