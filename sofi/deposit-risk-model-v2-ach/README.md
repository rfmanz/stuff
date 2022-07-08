# deposit-risk-model-v2-ach
---

Deposit risk model v2 - ACH to evaluate the likelihood of an ACH deposit will be returned in 3 business days.

* Build environment: `make environment`
* Query and build data: `make all` -- TODO: not yet updated to this directory
* Activate virtual environment: `source activate deposit_v2`
* Model Training: `make train`

We will take the `config.json` file as the input to the scripts and results will be stored in artifacts. See `stdout` log for details.


--- 

### Install dependency

```
pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt
```

---

### Some data intricasy during processing.

Dataset processed currently has 4 chunks as of May 2021. To change the number of chunks, please modify the `chunk_size` field in the `config.json` file. The dataloader will group the data by id:`business_account_number`, process them, and make sure each group has in total of less than `chunk_size` transactions.

chunk_size is currently set to 10,000,000 with a m5.25xlarge machine

--- 

### Load full transactions data

Due to the size of our data, it is not adviced to load all columns into the memory for processing or model building. Please lead data as instructed in `how-to-load.ipynb`.

Or

```python
import json
import numpy as np
import pandas as pd
from rdsutils.datasets import DataLoader
from src.utils import get_data_dir

with open("config.json", "r") as f:
    config = json.load(f)
    
# the last stage for the ETL is "features", for this task
# labeling was carried out in "features" stage for technical debt reasons
# be careful of the relative location of base_path. It depends on whether you called the function
base_path = config["base_path"]
data_dir = get_data_dir(config, base_path, "features")

# modify the columns
cols = ["transaction_id", "business_account_number", "transaction_datetime", 
        "is_returned", "target_60d"]

dl = DataLoader(data_dir, columns=cols)
df = dl.get_full()
df.shape
```