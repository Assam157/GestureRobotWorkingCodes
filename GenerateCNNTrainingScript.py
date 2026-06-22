import pandas as pd
import numpy as np

df = pd.read_csv(
    "gesture_wrist_dataset.csv"
)

WINDOW = 10

samples = []

for i in range(
    len(df)-WINDOW
):

    history = []

    for j in range(WINDOW):

        history.extend([
            df.iloc[i+j]["x"],
            df.iloc[i+j]["y"],
            df.iloc[i+j]["z"]
        ])

    target = [

        df.iloc[i+WINDOW]["x"],
        df.iloc[i+WINDOW]["y"],
        df.iloc[i+WINDOW]["z"]
    ]

    samples.append(
        history + target
    )

columns = []

for i in range(WINDOW):

    columns += [
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ]

columns += [
    "target_x",
    "target_y",
    "target_z"
]

pd.DataFrame(
    samples,
    columns=columns
).to_csv(
    "cnn_training_dataset.csv",
    index=False
)
print("saved ")