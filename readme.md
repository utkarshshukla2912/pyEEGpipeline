## EEG Pipeline

```

├── dataset/
│   ├── class1/ eg happy
|   |   ├── datafile1.mat or datafile1.edf or datafile1.csv
|   |   ├── datafile2.mat or datafile2.edf or datafile2.csv
|   |   └── datafile3.mat or datafile3.edf or datafile3.csv
│   ├── class2/ eg sad
|   |   ├── datafile1.mat or datafile1.edf or datafile1.csv
|   |   ├── datafile2.mat or datafile2.edf or datafile2.csv
|   |   └── datafile3.mat or datafile3.edf or datafile3.csv
│   └── class3/ eg calm
|       ├── datafile1.mat or datafile1.edf or datafile1.csv
|       ├── datafile2.mat or datafile2.edf or datafile2.csv
|       └── datafile3.mat or datafile3.edf or datafile3.csv
|
└── objects/
    ├── preProcessedData/
    |   ├── datafile2.npy
    |   └── datafile2.npy
    └── preProcessedData/
        ├── channel_split_features.csv
        └── channel_specific.csv

```
