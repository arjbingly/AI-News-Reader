from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from transformers import pipeline
class ZeroShotClassifier:
    def __init__(self, model_name='facebook/bart-large-mnli'):
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline("zero-shot-classification",
                                   # model="facebook/bart-large-mnli",
                                   model="AyoubChLin/Bart-MNLI-CNN_news",
                                    )

    def classify(self, sequence, labels):
        # Create an empty list to store the results
        results = []

        return self.classifier(sequence, labels)

        # # Iterate through each label and classify the sequence
        # for label in labels:
        #     premise = sequence
        #     hypothesis = f'This example is {label}.'
        #
        #     # run through the model pre-trained on MNLI
        #     inputs = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first')
        #     logits = self.model(inputs)[0]
        #
        #     # we throw away "neutral" (dim 1) and take the probability of
        #     # "entailment" (2) as the probability of the label being true
        #     entail_contradiction_logits = logits[:, [0, 2]]
        #     probs = entail_contradiction_logits.softmax(dim=1)
        #     prob_label_is_true = probs[:, 1].item()  # Extracting a scalar value
        #
        #     # Append the result to the list
        #     results.append({"Label": label, "Probability": prob_label_is_true})
        #
        # # Create a DataFrame from the list of results
        # results_df = pd.DataFrame(results)
        #
        # return results_df

# # Example usage
# url = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'
# news = NewsArticle(url)
# sequence = news.article.text
#
# labels = ["movies", "politics", "Hollywood"]
#
# classifier_instance = ZeroShotClassifier()
# results = classifier_instance.classify(sequence, labels)
# print(results)

