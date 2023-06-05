import pandas as pd
import numpy as np


group_order = ["Amplitude", "Phase", "Amplitude and Phase", "Frequency"]
group_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 3,
    22: 3,
    23: 1,
}


def get_eval_predictions(model, eval_ds):
    labels = []
    preds = []
    snrs = []

    for d, l, s in eval_ds:
        preds += list(np.argmax(model.predict(d), axis=1))
        labels += list(l.numpy())
        snrs += list(s.numpy())

    df = pd.DataFrame({"Label": labels, "Prediction": preds, "SNR": snrs})
    return df


def evaluate_varying_snrs_accs(df):

    acc_arr = (
        df.groupby("SNR")
        .apply(lambda x: (x["Label"] == x["Prediction"]).sum() / len(x["Label"]))
        .reset_index(name="Accuracy")
    ).to_numpy()

    snrs = list(acc_arr[:, 0].astype(int).astype(str))
    accs = list(acc_arr[:, 1])

    return {s: a for s, a in zip(snrs, accs)}


def add_group_info(df):
    df["Group Label"] = df["Label"].apply(lambda x: group_dict[x])
    df["Group Prediction"] = df["Prediction"].apply(lambda x: group_dict[x])
    return df


def evaluate_group_accs(df):
    df = add_group_info(df)

    acc_arr = (
        df.groupby("Group Label")
        .apply(
            lambda x: (x["Group Label"] == x["Group Prediction"]).sum()
            / len(x["Group Label"])
        )
        .reset_index(name="Group Accuracy")
    ).to_numpy()

    groups = list(acc_arr[:, 0].astype(int))
    groups = [group_order[g] for g in groups]
    accs = list(acc_arr[:, 1])

    return {g: a for g, a in zip(groups, accs)}


def evaluate_group_varying_snr_accs(df):
    df = add_group_info(df)

    acc_df = (
        df.groupby(["Group Label", "SNR"])
        .apply(
            lambda x: (x["Group Label"] == x["Group Prediction"]).sum()
            / len(x["Group Label"])
        )
        .reset_index(name="Group Accuracy")
    )

    groups = np.unique(acc_df["Group Label"])

    results = {}
    for g in groups:
        g_df = acc_df[acc_df["Group Label"] == g]
        g_df = g_df.sort_values(by="SNR", ascending=True)
        arr = g_df.to_numpy()
        g_label = group_order[int(g)]
        snrs = list(arr[:, 1].astype(int).astype(str))
        accs = list(arr[:, 2])
        results[g_label] = {s: a for s, a in zip(snrs, accs)}

    return results
