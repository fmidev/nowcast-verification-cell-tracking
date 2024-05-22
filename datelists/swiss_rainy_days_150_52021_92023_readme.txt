# Data description

Creation date: 2023-11-22 16:11:19

The timestamps were split into 3 training blocks,
2.5 testing blocks and 0.5 validation blocks.

Filenames:

- Training: swiss_rainy_days_150_52021_92023_train.txt
- Testing: swiss_rainy_days_150_52021_92023_test.txt
- Validation: swiss_rainy_days_150_52021_92023_valid.txt

The lists were created with the following command:

```bash
python split_datelist_swiss_dataset.py \
    swiss_rainy_days_150_52021_92023.txt \
    --data-freq 5 \
    --n-train-blocks 3 \
    --n-test-blocks 2.5 \
    --n-valid-blocks 0.5
```

The following other arguments were used:

- FILENAME_TEMPLATE = /scratch/jritvane/tmp/rainrate/%Y/%m/%d/RZC%y%j%H%MVL.*
- BBOX = [0, 640, 0, 710]
- R_THR = 1.0
- PCT_THR = 1
- len_block = 48

