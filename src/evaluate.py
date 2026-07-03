import pandas as pd
from bert_score import BERTScorer


df = pd.read_csv("results/baseline_evaluation.csv")

predictions = df["Prediction"].to_list()
ground_truth = df["Ground_truth"].to_list()


# Load RoBERTa ONCE
scorer = BERTScorer(
    model_type="roberta-large",
    lang="en"
)


batch_size = 30

all_precision = []
all_recall = []
all_f1 = []


for i in range(0, len(predictions), batch_size):

    print(f"Processing batch {i}/{len(predictions)}")

    pred_batch = predictions[i:i + batch_size]
    truth_batch = ground_truth[i:i + batch_size]


    P, R, F1 = scorer.score(
        pred_batch,
        truth_batch
    )


    all_precision.extend(P.tolist())
    all_recall.extend(R.tolist())
    all_f1.extend(F1.tolist())


print("BERTScore calculation complete")


metrics = {
    "precision": sum(all_precision) / len(all_precision),
    "recall": sum(all_recall) / len(all_recall),
    "f1": sum(all_f1) / len(all_f1)
}


pd.DataFrame([metrics]).to_csv(
    "results/baseline_metrics.csv",
    index=False
)


print(f"Baseline BERTScore F1: {metrics['f1']}")